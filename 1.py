import yaml
from uuid import uuid4

def generate_rasa_files(qa_pairs, output_dir="data"):
    # Generate responses.yml
    responses = {"version": "3.1", "responses": {}}
    
    for idx, pair in enumerate(qa_pairs):
        faq_id = f"faq_{uuid4().hex[:6]}"  # Unique ID for each FAQ
        responses["responses"][f"utter_faq/{faq_id}"] = [{"text": pair["answer"]}]
    
    # Save responses.yml
    with open(f"{output_dir}/responses.yml", "w") as f:
        yaml.dump(responses, f, default_flow_style=False)

    # Generate faq.yml (retrieval intent config)
    faq_config = {
        "version": "3.1",
        "nlu": [{
            "intent": "ask_faq",
            "examples": "\n".join([f"- {q['question']}" for q in qa_pairs])
        }],
        "retrieval_intents": ["faq"]
    }
    
    with open(f"{output_dir}/faq.yml", "w") as f:
        yaml.dump(faq_config, f, default_flow_style=False)

# Usage
qa_pairs = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "What is the population of Paris?", "answer": "The population of Paris is approximately 2.1 million."}
]
generate_rasa_files(qa_pairs)
