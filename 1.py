import pdfplumber
import re

def parse_qa_from_pdf(pdf_path):
    """
    Parse questions and answers from a PDF with Q: and A: format markers
    Returns a list of dictionaries with keys 'question' and 'answer'
    """
    qa_pairs = []
    
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        
        # Extract text from all pages
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
        
        # Split text into QA blocks using lookbehind for Q: or start of string
        qa_blocks = re.split(r'(?<=\n)Q:|Q:', full_text)
        
        for block in qa_blocks[1:]:  # Skip first element if empty
            # Split into question and answer parts
            parts = re.split(r'\nA:', block, maxsplit=1)
            
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()
                
                # Clean up any remaining newlines
                question = ' '.join(question.split())
                answer = ' '.join(answer.split())
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer
                })
    
    return qa_pairs

# Usage example
pdf_path = "your_file.pdf"
qa_data = parse_qa_from_pdf(pdf_path)

# Print results
for idx, pair in enumerate(qa_data, 1):
    print(f"Pair {idx}:")
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer']}\n")
