intent_name = "generic"
patterns = {
    "greet": ["hi", "hello", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "goodbye", "see you", "take care"],
    "mood_great": ["i'm happy", "feeling good", "doing great"],
    "mood_unhappy": ["i'm sad", "not feeling well", "bad mood"],
    "bot_challenge": ["are you a bot", "who made you", "are you real"],
    "affirm": ["yes", "sure", "correct", "absolutely"],
    "deny": ["no", "never", "wrong"],
    "end_session": ["end chat", "exit", "goodbye session"],
    "skills_list": ["what can you do", "help options", "show features"]
}

class TaskHandler:
    def start(self, sub_intent):
        from utils import load_responses
        responses = load_responses("generic")
        return responses.get(sub_intent, ["I'm not sure how to respond to that."])[0]

    def handle_form_submission(self, form_data):
        return "This action does not require form submission."


{
    "greet": ["Hello!", "Hey there!", "Hi, how can I assist you?"],
    "goodbye": ["Goodbye!", "See you soon!", "Take care!"],
    "mood_great": ["That's awesome!", "Glad to hear that!"],
    "mood_unhappy": ["I'm here if you need to talk.", "Hope things get better soon."],
    "bot_challenge": ["Yes, I am a bot, but I'm here to help!", "I'm an AI assistant."],
    "affirm": ["Okay!", "Sounds good!", "Alright."],
    "deny": ["No worries!", "Understood.", "Alright, let me know if you need help."],
    "end_session": ["Ending session now. Goodbye!", "Take care!"],
    "skills_list": ["I can help with ticket creation, checking case details, and more."]
}



import re
from task_registry import TASKS

class IntentClassifier:
    def __init__(self):
        self.patterns = {}
        for intent, task in TASKS.items():
            if intent == "generic":
                for sub_intent, patterns in task.patterns.items():
                    self.patterns[sub_intent] = [re.compile(p, re.IGNORECASE) for p in patterns]
            else:
                self.patterns[intent] = [re.compile(p, re.IGNORECASE) for p in task.patterns]

    def classify(self, message):
        for intent, regexes in self.patterns.items():
            if any(r.search(message) for r in regexes):
                return intent
        return "fallback"





