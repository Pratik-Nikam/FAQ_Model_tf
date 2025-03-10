# Import required libraries
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict
from tensorflow.keras.models import load_model
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import random

# Load your Keras/TensorFlow models (update paths accordingly)
classification_model = load_model('path/to/classification_model.h5')
disposition_model = load_model('path/to/disposition_model.h5')
resolution_model = load_model('path/to/resolution_model.h5')

# Define ML models as LangGraph tools
@tool
def classify_issue(description: str) -> str:
    # Preprocess and predict using classification_model
    # Replace with actual inference logic
    return "hardware_failure"  # Example output

@tool
def determine_disposition(issue_type: str, extra_data: dict = None) -> str:
    # Preprocess and predict using disposition_model with issue_type and extra_data
    # Replace with actual inference logic
    return "support_staff_needed"  # Example output

@tool
def generate_resolution(disposition: str, extra_data: dict = None) -> str:
    # Preprocess and predict using resolution_model with disposition and extra_data
    # Replace with actual inference logic
    return "Escalate to Team X"  # Example output

tools = [classify_issue, determine_disposition, generate_resolution]
tool_node = ToolNode(tools)

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Conversation history
    issue_type: str  # Output from Issue Classifier
    disposition: str  # Output from Disposition Model
    resolution: str  # Output from Resolution Model
    collected_data: dict  # Data fetched from external systems
    decision_log: list  # Log of decisions for learning (e.g., data collection choices)

# Bandit algorithm for learning
class BanditLearner:
    def __init__(self, options: list):
        self.options = options
        self.values = {opt: 0.5 for opt in options}  # Initial success rates
        self.counts = {opt: 0 for opt in options}  # Attempts
        self.epsilon = 0.1  # Exploration rate

    def choose_option(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.options)  # Explore
        return max(self.values, key=self.values.get)  # Exploit best option

    def update(self, option: str, reward: float):
        self.counts[option] += 1
        n = self.counts[option]
        value = self.values[option]
        self.values[option] = value + (reward - value) / n

# Initialize bandit learner
data_bandit = BanditLearner(options=["minimal", "full"])

# Build the LangGraph workflow
workflow = StateGraph(AgentState)

# Node 1: Classify the issue
def classify_node(state: AgentState) -> dict:
    description = state["messages"][-1].content
    issue_type = classify_issue.invoke(description)
    return {"issue_type": issue_type, "messages": [HumanMessage(content=f"Issue: {issue_type}")]}

# Node 2: Collect data and determine disposition
def disposition_node(state: AgentState) -> dict:
    # Decide how much data to collect using bandit algorithm
    data_choice = data_bandit.choose_option()
    if data_choice == "minimal":
        collected_data = {"user_id": "123"}  # Example: minimal data from system A
    else:
        collected_data = {"user_id": "123", "device_info": "printer_model_X"}  # Full data
    disposition = determine_disposition.invoke(state["issue_type"], collected_data)
    return {
        "disposition": disposition,
        "collected_data": collected_data,
        "decision_log": state.get("decision_log", []) + [{"step": "disposition", "choice": data_choice}]
    }

# Node 3: Collect more data and generate resolution
def resolution_node(state: AgentState) -> dict:
    data_choice = data_bandit.choose_option()
    if data_choice == "minimal":
        extra_data = state["collected_data"] | {"priority": "high"}
    else:
        extra_data = state["collected_data"] | {"priority": "high", "team_status": "available"}
    resolution = generate_resolution.invoke(state["disposition"], extra_data)
    return {
        "resolution": resolution,
        "collected_data": extra_data,
        "decision_log": state["decision_log"] + [{"step": "resolution", "choice": data_choice}],
        "messages": [HumanMessage(content=resolution)]
    }

# Add nodes to workflow
workflow.add_node("classify", classify_node)
workflow.add_node("disposition", disposition_node)
workflow.add_node("resolution", resolution_node)

# Define the flow
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "disposition")
workflow.add_edge("disposition", "resolution")
workflow.add_edge("resolution", END)

# Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Rasa custom actions
class ActionAgentResponse(Action):
    def name(self) -> str:
        return "action_agent_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        description = tracker.get_slot("description") or tracker.latest_message["text"]
        config = {"configurable": {"thread_id": tracker.sender_id}}
        result = graph.invoke({"messages": [HumanMessage(content=description)]}, config)
        resolution = result["resolution"]
        dispatcher.utter_message(text=f"Resolution: {resolution}. Was this helpful? (Yes/No)")
        return []

class ActionCollectFeedback(Action):
    def name(self) -> str:
        return "action_collect_feedback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        feedback = tracker.latest_message["text"].lower()
        reward = 1.0 if "yes" in feedback else 0.0
        state = graph.get_state({"configurable": {"thread_id": tracker.sender_id}}).values
        last_decision = state["decision_log"][-1]["choice"]
        data_bandit.update(last_decision, reward)
        dispatcher.utter_message(text="Thanks for the feedback!")
        return []

##############
# Rasa Configuration (domain.yml)
"""
version: "3.1"

intents:
  - report_issue
  - provide_details
  - not_satisfied
  - affirm
  - deny

entities:
  - description

slots:
  description:
    type: text
    influence_conversation: true
    mappings:
      - type: from_text
        intent: report_issue
        conditions:
          - active_loop: null
      - type: from_text
        intent: provide_details
  feedback:
    type: text
    influence_conversation: false
    mappings:
      - type: from_intent
        intent: affirm
        value: "yes"
      - type: from_intent
        intent: deny
        value: "no"

responses:
  utter_ask_more_details:
    - text: "Could you provide more details about the issue? This will help me assist you better."
  utter_present_disposition:
    - text: "It seems like your issue requires {disposition}. Does this sound right?"
  utter_present_resolution:
    - text: "Here’s a resolution: {resolution}. Was this helpful? (Yes/No)"
  utter_try_again:
    - text: "Sorry that didn’t help. Let me try again with more information."
  utter_thanks_feedback:
    - text: "Thanks for the feedback!"

actions:
  - action_agent_response
  - action_collect_feedback

session_config:
  session_expiration_time: 60
  carry_over_slots_to_new_session: true
"""

##############
# Rasa Rules (rules.yml)
"""
version: "3.1"

rules:
  - rule: Handle initial issue report
    steps:
      - intent: report_issue
      - action: action_agent_response

  - rule: Ask for more details if needed
    steps:
      - action: action_agent_response
      - slot_was_set:
          - description: "more_details_needed"
      - action: utter_ask_more_details

  - rule: Handle additional details
    steps:
      - intent: provide_details
      - action: action_agent_response

  - rule: Handle dissatisfaction
    steps:
      - intent: not_satisfied
      - action: action_agent_response

  - rule: Collect feedback after resolution
    steps:
      - action: action_agent_response
      - slot_was_set:
          - resolution: not_null
      - action: utter_present_resolution
      - intent: affirm
        or:
        - intent: deny
      - action: action_collect_feedback
"""

##############
# Rasa NLU (nlu.yml)
"""
version: "3.1"

nlu:
  - intent: report_issue
    examples: |
      - I’m facing a blah blah issue
      - My printer isn’t working
  - intent: provide_details
    examples: |
      - It’s making a weird noise
      - The screen is blank
  - intent: not_satisfied
    examples: |
      - I’m not satisfied
      - That didn’t help
  - intent: affirm
    examples: |
      - Yes
      - That worked
  - intent: deny
    examples: |
      - No
      - Not helpful
"""

##############
# LangGraph Agent Code (agent.py)
# Import required libraries
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict
from tensorflow.keras.models import load_model
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import random

# Load your Keras/TensorFlow models (update paths accordingly)
classification_model = load_model('path/to/classification_model.h5')
disposition_model = load_model('path/to/disposition_model.h5')
resolution_model = load_model('path/to/resolution_model.h5')

# Define ML models as LangGraph tools
@tool
def classify_issue(description: str) -> str:
    # Preprocess and predict using classification_model
    # Replace with actual inference logic
    # Simulate confidence check (e.g., < 5 chars = unclear)
    if len(description) < 5:
        return "unclear"
    return "hardware_failure"  # Example output

@tool
def determine_disposition(issue_type: str, extra_data: dict = None) -> str:
    # Preprocess and predict using disposition_model
    return "support_staff_needed"  # Example output

@tool
def generate_resolution(disposition: str, extra_data: dict = None) -> str:
    # Preprocess and predict using resolution_model
    return "Escalate to Team X" + (f" (FAQ: example.com/faq)" if extra_data.get("level", 1) > 1 else "")

tools = [classify_issue, determine_disposition, generate_resolution]
tool_node = ToolNode(tools)

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    issue_type: str
    disposition: str
    resolution: str
    collected_data: dict
    decision_log: list
    attempt_level: int  # Tracks incremental attempts

# Bandit algorithm for learning
class BanditLearner:
    def __init__(self, options: list):
        self.options = options
        self.values = {opt: 0.5 for opt in options}
        self.counts = {opt: 0 for opt in options}
        self.epsilon = 0.1

    def choose_option(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.options)
        return max(self.values, key=self.values.get)

    def update(self, option: str, reward: float):
        self.counts[option] += 1
        n = self.counts[option]
        value = self.values[option]
        self.values[option] = value + (reward - value) / n

data_bandit = BanditLearner(options=["minimal", "full"])

# Build the LangGraph workflow
workflow = StateGraph(AgentState)

def classify_node(state: AgentState) -> dict:
    description = state["messages"][-1].content
    issue_type = classify_issue.invoke(description)
    if issue_type == "unclear":
        return {"issue_type": "more_details_needed", "messages": [HumanMessage(content="more_details_needed")]}
    return {"issue_type": issue_type, "attempt_level": state.get("attempt_level", 1)}

def disposition_node(state: AgentState) -> dict:
    if state["issue_type"] == "more_details_needed":
        return state
    data_choice = data_bandit.choose_option()
    if data_choice == "minimal":
        collected_data = {"user_id": "123"}
    else:
        collected_data = {"user_id": "123", "device_info": "printer_model_X"}
    disposition = determine_disposition.invoke(state["issue_type"], collected_data)
    return {
        "disposition": disposition,
        "collected_data": collected_data,
        "decision_log": state.get("decision_log", []) + [{"step": "disposition", "choice": data_choice}]
    }

def resolution_node(state: AgentState) -> dict:
    if state["issue_type"] == "more_details_needed":
        return state
    data_choice = data_bandit.choose_option()
    attempt_level = state.get("attempt_level", 1)
    if data_choice == "minimal":
        extra_data = state["collected_data"] | {"priority": "high", "level": attempt_level}
    else:
        extra_data = state["collected_data"] | {"priority": "high", "team_status": "available", "level": attempt_level}
    resolution = generate_resolution.invoke(state["disposition"], extra_data)
    return {
        "resolution": resolution,
        "collected_data": extra_data,
        "decision_log": state["decision_log"] + [{"step": "resolution", "choice": data_choice}]
    }

def handle_dissatisfaction(state: AgentState) -> dict:
    attempt_level = state.get("attempt_level", 1) + 1
    return {"attempt_level": attempt_level, "resolution": None, "messages": [HumanMessage(content="Trying again...")]}

workflow.add_node("classify", classify_node)
workflow.add_node("disposition", disposition_node)
workflow.add_node("resolution", resolution_node)
workflow.add_node("dissatisfaction", handle_dissatisfaction)

workflow.add_edge(START, "classify")
workflow.add_edge("classify", "disposition")
workflow.add_edge("disposition", "resolution")
workflow.add_conditional_edges(
    "resolution",
    lambda state: "dissatisfaction" if "not satisfied" in state["messages"][-1].content.lower() else END
)
workflow.add_edge("dissatisfaction", "resolution")

# Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Rasa custom actions
class ActionAgentResponse(Action):
    def name(self) -> str:
        return "action_agent_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        description = tracker.get_slot("description") or tracker.latest_message["text"]
        config = {"configurable": {"thread_id": tracker.sender_id}}
        result = graph.invoke({"messages": [HumanMessage(content=description)]}, config)
        
        if result["issue_type"] == "more_details_needed":
            dispatcher.utter_message(response="utter_ask_more_details")
            return []
        elif result["resolution"]:
            dispatcher.utter_message(response="utter_present_resolution", resolution=result["resolution"])
        elif result["disposition"]:
            dispatcher.utter_message(response="utter_present_disposition", disposition=result["disposition"])
        return []

class ActionCollectFeedback(Action):
    def name(self) -> str:
        return "action_collect_feedback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        feedback = tracker.get_slot("feedback")
        reward = 1.0 if feedback == "yes" else 0.0
        state = graph.get_state({"configurable": {"thread_id": tracker.sender_id}}).values
        last_decision = state["decision_log"][-1]["choice"]
        data_bandit.update(last_decision, reward)
        dispatcher.utter_message(response="utter_thanks_feedback")
        return []




import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
import pandas as pd

# Example data
data = {'problem': ['hello world', 'test case'], 'resolution': ['goodbye world', 'pass test']}
df = pd.DataFrame(data)

# Lowercase to avoid case sensitivity issues
df['problem'] = df['problem'].str.lower()
df['resolution'] = df['resolution'].str.lower()

# Tokenizer
all_text = df['problem'].tolist() + df['resolution'].tolist()
special_tokens = ['<start>', '<end>']
tokenizer = Tokenizer(oov_token='<UNK>')
tokenizer.fit_on_texts(all_text + special_tokens)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Prepare sequences
df['resolution_full'] = df['resolution'].apply(lambda x: '<start> ' + x + ' <end>')
problem_sequences = pad_sequences(tokenizer.texts_to_sequences(df['problem']), maxlen=10, padding='post')
resolution_sequences = pad_sequences(tokenizer.texts_to_sequences(df['resolution_full']), maxlen=10, padding='post')

# Check indices
print("Max index in problem_sequences:", np.max(problem_sequences))
print("Max index in resolution_sequences:", np.max(resolution_sequences))
assert np.max(problem_sequences) < vocab_size, "Out-of-range indices in problem_sequences"
assert np.max(resolution_sequences) < vocab_size, "Out-of-range indices in resolution_sequences"

# Model parameters
embedding_dim = 256
lstm_units = 512
max_seq_len = 10

# Encoder
encoder_inputs = Input(shape=(max_seq_len,))
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_seq_len-1,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units, return_sequences=True)(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')(decoder_lstm)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_dense)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training data
decoder_input_data = resolution_sequences[:, :-1]
decoder_target_data = resolution_sequences[:, 1:]

# Train
model.fit(
    [problem_sequences, decoder_input_data],
    decoder_target_data,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)
