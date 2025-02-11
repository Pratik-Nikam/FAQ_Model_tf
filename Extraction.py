
import fitz  # PyMuPDF

def extract_qna_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    qna_list = []
    question, answer = None, None

    for page in doc:
        text = page.get_text("text")  # Extract text from page
        lines = text.split("\n")  # Split text into lines
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith("q:"):
                # Save previous Q&A if available
                if question and answer:
                    qna_list.append((question, answer))
                question = line  # Store new question
                answer = None  # Reset answer
            elif line.lower().startswith("a:"):
                answer = line  # Store answer

    # Append last Q&A pair
    if question and answer:
        qna_list.append((question, answer))

    return qna_list

# Example usage:
pdf_path = "your_faq.pdf"  # Replace with your PDF file path
qna_pairs = extract_qna_from_pdf(pdf_path)

# Print extracted Q&A
for q, a in qna_pairs:
    print(q)
    print(a)
    print("-" * 50)

# Save results to a file (optional)
with open("extracted_qna.txt", "w", encoding="utf-8") as f:
    for q, a in qna_pairs:
        f.write(q + "\n" + a + "\n\n")

Features:

Extracts questions (lines starting with Q:) and answers (A:).

Ignores unnecessary text.

Stores extracted Q&A pairs in a list.

Saves results to a .txt file for easy access.


Next Steps:

Run this script on your FAQ PDF.

If needed, upload the file here, and I’ll process it for you.


