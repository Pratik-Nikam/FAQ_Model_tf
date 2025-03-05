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

# To run the agent, you would typically integrate it with Rasa's action server
# Start Rasa with: rasa run actions
# Interact via Rasa shell or API
