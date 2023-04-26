import os
import threading

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
from train_model import load_dataset, train, get_stats

app = Flask(__name__)
app.secret_key = "mysecretkey"

# Initialize SocketIO
socketio = SocketIO(app)

# Configure LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create a simple User model
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# User loader for LoginManager
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Login view
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == '1' and password == '1':
            user = User(1)
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

# Logout view
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Index view
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Training view
@app.route('/train', methods=['POST', 'GET'])
@login_required
def train_view():
    if request.method == 'POST':
        dataset_name = request.form['dataset']
        algorithm = request.form['algorithm']
    else:
        dataset_name = request.args.get('dataset')
        algorithm = request.args.get('algorithm')

    return render_template('train.html', dataset=dataset_name, algorithm=algorithm, acc=0.8, auc=0.85)  # Update acc and auc with actual values

# New route to get dataset stats
@app.route('/get_stats', methods=['POST'])
@login_required
def get_stats_view():
    dataset_name = request.form['dataset']
    dataset = load_dataset(dataset_name)
    stats = get_stats(dataset)

    return jsonify(stats)

# New route to start training
@app.route('/start_train', methods=['POST'])
@login_required
def start_train_view():
    dataset_name = request.form['dataset']
    algorithm = request.form['algorithm']
    custom_code = request.form.get('custom_code', None)

    thread = threading.Thread(target=train_and_emit_progress, args=(dataset_name, algorithm, custom_code))
    thread.start()
    return redirect(url_for('train_view', dataset=dataset_name, algorithm=algorithm))

# Training function that emits progress updates
def train_and_emit_progress(dataset_name, algorithm, custom_code):
    dataset = load_dataset(dataset_name)
    model = train(dataset, algorithm, socketio, custom_code)


if __name__ == '__main__':
    socketio.run(app)