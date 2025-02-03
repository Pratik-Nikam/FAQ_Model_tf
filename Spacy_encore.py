import pandas as pd
import spacy
from spacy.training import Example
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess data from Excel
def load_data(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Convert Flavor column to list of labels (assuming labels are comma-separated)
    df['Flavor'] = df['Flavor'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # Drop rows with missing data
    df.dropna(subset=['Description', 'Flavor'], inplace=True)

    return df

# Step 2: Convert data to spaCy format
def prepare_spacy_data(df):
    train_data = []

    for index, row in df.iterrows():
        text = row['Description']
        labels = {label: 1 for label in row['Flavor']}  # Convert labels to dictionary
        train_data.append((text, {"cats": labels}))

    return train_data

# Step 3: Train the spaCy model
def train_spacy_model(train_data, val_data, n_epochs=10):
    # Load the pre-trained model
    nlp = spacy.load("en_core_web_md")

    # Add text categorizer to the pipeline
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    # Add labels to the text categorizer
    all_labels = set()
    for _, annotations in train_data:
        all_labels.update(annotations['cats'].keys())
    for label in all_labels:
        textcat.add_label(label)

    # Convert training data to spaCy Example objects
    train_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in train_data]
    val_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in val_data]

    # Disable other pipeline components during training
    with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != "textcat"]):
        optimizer = nlp.begin_training()
        for epoch in range(n_epochs):
            losses = {}
            for example in train_examples:
                nlp.update([example], drop=0.2, losses=losses, sgd=optimizer)
            print(f"Epoch {epoch + 1}, Loss: {losses['textcat']}")

    return nlp

# Step 4: Evaluate the model
def evaluate_model(nlp, val_data):
    val_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in val_data]
    scores = nlp.evaluate(val_examples)
    print("Evaluation Scores:", scores)

# Step 5: Save the model
def save_model(nlp, model_path):
    nlp.to_disk(model_path)
    print(f"Model saved to {model_path}")

# Main function
def main():
    # File paths
    excel_file_path = "your_file.xlsx"  # Replace with your Excel file path
    model_save_path = "textcat_model"  # Path to save the trained model

    # Step 1: Load and preprocess data
    df = load_data(excel_file_path)

    # Step 2: Prepare spaCy data
    train_data = prepare_spacy_data(df)

    # Step 3: Split data into training and validation sets
    train, val = train_test_split(train_data, test_size=0.2, random_state=42)

    # Step 4: Train the model
    nlp = train_spacy_model(train, val, n_epochs=10)

    # Step 5: Evaluate the model
    evaluate_model(nlp, val)

    # Step 6: Save the model
    save_model(nlp, model_save_path)

    # Test the model on a sample text
    test_text = "This is a sample description."
    doc = nlp(test_text)
    print("Predicted Labels:", doc.cats)

if __name__ == "__main__":
    main()
