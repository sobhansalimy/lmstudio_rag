import os
import fitz  # PyMuPDF
from openai import OpenAI

# Function to extract text from all PDFs in the 'pdfs' directory
def extract_text_from_pdfs(pdf_directory):
    text_data = ""
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            doc = fitz.open(filepath)
            for page in doc:
                text_data += page.get_text()
    return text_data

# Extract text from PDF files
pdf_directory = "pdfs"
pdf_text = extract_text_from_pdfs(pdf_directory)

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Function to ask questions based on extracted PDF text
def ask_question_about_pdfs(question):
    messages = [
        {"role": "system", "content": "You are an expert in analyzing PDF documents."},
        {"role": "user", "content": f"Here is the content of the PDFs: {pdf_text}"},
        {"role": "user", "content": question}
    ]
    
    completion = client.chat.completions.create(
        model="local-model",  # this field is currently unused
        messages=messages,
        temperature=0.7,
    )

    return completion.choices[0].message.content

# Interactive Q&A loop
print("Interactive PDF Q&A")
print("Type 'exit' to quit.")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    answer = ask_question_about_pdfs(question)
    print(f"AI: {answer}")
