import eventlet
eventlet.monkey_patch()  # Enables non-blocking behavior for Flask-SocketIO

from flask import Flask, request
from flask_socketio import SocketIO, emit

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize SocketIO with eventlet for high concurrency
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# In-memory dictionary to track user connections
users = {}

@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    sid = request.sid  # Unique session ID assigned by Flask-SocketIO
    users[sid] = {'sid': sid}
    print(f"Client connected: {sid}")
    emit('bot_response', {'text': 'Hello! You are now connected to the chatbot.'})

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle messages from the client"""
    sid = request.sid
    message = data.get('text', '').strip()

    if not message:
        return

    print(f"Received message from {sid}: {message}")

    # Process chatbot logic (simple echo here, replace with AI/DB logic)
    response_text = process_message(message)

    # Send response to the specific user
    emit('bot_response', {'text': response_text}, to=sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    sid = request.sid
    if sid in users:
        del users[sid]
    print(f"Client disconnected: {sid}")

def process_message(msg):
    """Simple chatbot logic - Echoes back the message (Replace with AI logic)"""
    return f"Chatbot: {msg}"

if __name__ == "__main__":
    print("Starting Flask-SocketIO chatbot server...")
    socketio.run(app, host="0.0.0.0", port=5000)
