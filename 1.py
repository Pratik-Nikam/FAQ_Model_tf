import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

from task_registry import load_tasks
load_tasks()

from sockets import setup_sockets
setup_sockets(socketio)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

class Config:
    SECRET_KEY = 'your_secret_key'


import json
import os

def load_adaptive_card(template_name):
    try:
        with open(f"templates/{template_name}.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading card: {template_name}", e)
        return {}

def load_responses(category):
    try:
        with open(f"responses/{category}.json", "r") as f:
            return json.load(f)
    except Exception:
        return []

import os
import importlib
import sys

TASKS = {}

def register_task(intent_name, task_instance):
    TASKS[intent_name] = task_instance

def get_task(intent_name):
    return TASKS.get(intent_name)

def load_tasks():
    sys.path.insert(0, "tasks")
    for file in os.listdir("tasks"):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            module = importlib.import_module(module_name)
            if hasattr(module, "intent_name") and hasattr(module, "patterns") and hasattr(module, "TaskHandler"):
                instance = module.TaskHandler()
                instance.intent_name = module.intent_name
                instance.patterns = module.patterns
                register_task(intent_name=module.intent_name, task_instance=




import re
from task_registry import TASKS

class IntentClassifier:
    def __init__(self):
        self.patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in task.patterns]
            for intent, task in TASKS.items()
        }

    def classify(self, message):
        for intent, regexes in self.patterns.items():
            if any(r.search(message) for r in regexes):
                return intent
        return "greeting"


from flask import request
from flask_socketio import emit
import random
from utils import load_adaptive_card, load_responses
from task_registry import get_task
from intent_classifier import IntentClassifier

classifier = IntentClassifier()
user_sessions = {}
greeting_responses = load_responses("greetings")

def setup_sockets(socketio):

    @socketio.on('connect')
    def handle_connect():
        print(f"Connected: {request.sid}")

    @socketio.on('session_request')
    def handle_session_request(data):
        sid = request.sid
        display_name = data.get("common_authenticated_user_display_name", "Guest")
        user_sessions[sid] = {"display_name": display_name, "state": None}

        emit("session_confirm", {"sid": sid}, to=sid)
        emit("bot_uttered", {"text": f"Hey {display_name}, how can I help you today?"}, to=sid)
        emit("bot_uttered", {"adaptive_card": load_adaptive_card("welcome")}, to=sid)

    @socketio.on('user_uttered')
    def handle_user_uttered(data):
        sid = request.sid
        message = data.get("message", "")
        session = user_sessions.get(sid, {})

        if session.get("state") == "awaiting_confirmation":
            if message.lower() in ["yes", "y"]:
                task = get_task(session["intent"])
                emit("bot_uttered", {"adaptive_card": load_adaptive_card(task.form_template())}, to=sid)
                session["state"] = "form_displayed"
            else:
                emit("bot_uttered", {"adaptive_card": load_adaptive_card("welcome")}, to=sid)
                session["state"] = None
            return

        if session.get("state") == "form_displayed":
            task = get_task(session["intent"])
            response = task.handle_form_submission(data.get("form_data", {}))
            emit("bot_uttered", {"text": response}, to=sid)
            session["state"] = None
            return

        intent = classifier.classify(message)
        task = get_task(intent)

        if task:
            emit("bot_uttered", {"text": task.confirmation_prompt() + " (yes/no)"}, to=sid)
            session["intent"] = intent
            session["state"] = "awaiting_confirmation"
        elif intent == "greeting":
            emit("bot_uttered", {"text": random.choice(greeting_responses)}, to=sid)
        else:
            emit("bot_uttered", {"adaptive_card": load_adaptive_card("welcome")}, to=sid)

    @socketio.on('disconnect')
    def handle_disconnect():
        user_sessions.pop(request.sid, None)
        print(f"Disconnected: {request.sid}")


intent_name = "create_ticket"
patterns = ["create.*ticket", "open.*ticket", "report.*issue"]

class TaskHandler:
    def confirmation_prompt(self):
        return "You want to create a new support ticket. Shall I proceed?"

    def form_template(self):
        return "create_ticket"

    def start(self):
        return self.confirmation_prompt(), True

    def handle_form_submission(self, form_data):
        print("Form data received:", form_data)
        return "✅ Ticket submitted! We will get back to you shortly."


{
  "type": "AdaptiveCard",
  "body": [
    {
      "type": "TextBlock",
      "text": "Welcome! Choose a task or type your request.",
      "weight": "Bolder",
      "size": "Medium"
    }
  ],
  "actions": [
    {
      "type": "Action.Submit",
      "title": "Create Ticket",
      "data": { "task": "create_ticket" }
    }
  ],
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.2"
}


{
  "type": "AdaptiveCard",
  "body": [
    {
      "type": "TextBlock",
      "text": "Create a Ticket",
      "weight": "Bolder",
      "size": "Medium"
    },
    {
      "type": "Input.Text",
      "id": "title",
      "placeholder": "Enter ticket title"
    },
    {
      "type": "Input.Text",
      "id": "description",
      "placeholder": "Describe your issue",
      "isMultiline": true
    }
  ],
  "actions": [
    {
      "type": "Action.Submit",
      "title": "Submit",
      "data": { "submit": true }
    }
  ],
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.2"
}


[
  "Hi there! How can I help you?",
  "Hello! Need assistance?",
  "Hey! What can I do for you?",
  "Greetings! How may I assist you today?"
]



else:
    # Fallback: unknown input
    emit("bot_uttered", {"text": "Sorry, I didn't understand that. Please try again or choose an option below."}, to=sid)

    # Dynamically show available tasks
    from task_registry import TASKS
    task_buttons = [
        {
            "type": "Action.Submit",
            "title": task.intent_name.replace("_", " ").title(),
            "data": { "task": task.intent_name }
        }
        for task in TASKS.values()
    ]

    fallback_card = {
        "type": "AdaptiveCard",
        "body": [
            {
                "type": "TextBlock",
                "text": "Here are some things I can help with:",
                "weight": "Bolder",
                "size": "Medium"
            },
            {
                "type": "TextBlock",
                "text": "Or rephrase your question if it's something else."
            }
        ],
        "actions": task_buttons,
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.2"
    }

    emit("bot_uttered", {"adaptive_card": fallback_card}, to=sid)








