import os
import fitz  # PyMuPDF
from openai import OpenAI
from TTS.api import TTS
import pygame
import threading

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

# Initialize TTS with the new model
tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch")

# Create a threading lock
file_lock = threading.Lock()

# List to keep track of generated TTS files
tts_files = []
response_count = 0

# Function to convert text to speech and play it
def speak(text):
    global response_count

    thread_id = threading.get_ident()
    tts_file = f"response_{thread_id}.wav"
    
    # Synthesize the text to speech
    tts.tts_to_file(text=text, file_path=tts_file)
    
    # Acquire the lock before playing the sound
    with file_lock:
        pygame.mixer.init()
        pygame.mixer.music.load(tts_file)
        pygame.mixer.music.play()
        
        # Wait until the sound finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        # Keep track of the TTS files and response count
        tts_files.append(tts_file)
        response_count += 1

        # Remove old files if there are more than 2 responses
        if response_count > 2:
            old_file = tts_files.pop(0)
            if os.path.exists(old_file):
                os.remove(old_file)

# Function to handle speaking in a separate thread
def speak_in_thread(text):
    thread = threading.Thread(target=speak, args=(text,))
    thread.start()

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
        speak_in_thread(answer)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:  # Catch all other exceptions, including OpenAI errors
        print(f"An error occurred: {e}")
