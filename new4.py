import os
import fitz  # PyMuPDF
from openai import OpenAI
from TTS.api import TTS
import pygame

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

# Ensure pdf_text is not empty
if not pdf_text.strip():
    raise ValueError("No text extracted from PDFs. Ensure the 'pdfs' directory contains valid PDF files with text.")

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Function to ask questions based on extracted PDF text
def ask_question_about_pdfs(question):
    # Ensure the question is not empty
    if not question.strip():
        raise ValueError("The question cannot be empty.")

    messages = [
        {"role": "system", "content": "You are an expert in analyzing PDF documents."},
        {"role": "user", "content": f"Here is the content of the PDFs: {pdf_text}"},
        {"role": "user", "content": question}
    ]
    
    # OpenAI completion with streaming
    response = client.chat.completions.create(
        model="local-model",
        messages=messages,
        temperature=0.7,
        stream=True  # Enable streaming
    )

    full_answer = ""
    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
            full_answer += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end='', flush=True)

    return full_answer

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch")

# Function to convert text to speech and play it
def speak(text):
    tts_file = "response.wav"
    tts.tts_to_file(text=text, file_path=tts_file)
    pygame.mixer.init()
    pygame.mixer.music.load(tts_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Interactive Q&A loop
print("Interactive PDF Q&A")
print("Type 'exit' to quit.")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    try:
        answer = ask_question_about_pdfs(question)
        print()  # For a new line after the full answer
        speak(answer)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:  # Catch all other exceptions, including OpenAI errors
        print(f"An error occurred: {e}")
