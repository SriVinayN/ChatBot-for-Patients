# Install required packages if not already installed
# pip install trafilatura sentence-transformers faiss-cpu transformers textblob pymupdf openai

import trafilatura
import faiss
import requests
import fitz  # PyMuPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import re
import openai
import os

# -------------------------
# Set your OpenAI API key here or via environment variable OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY") or "Replace_with_your_key"

# -------------------------
# STEP 1: Helper Functions
# -------------------------
def extract_pdf_text_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Could not download PDF from {url}")
        return None
    pdf_data = BytesIO(response.content)
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def extract_text_from_local_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return "".join([page.get_text() for page in doc])
    else:
        print(f"Unsupported file type: {file_path}")
        return None

# -------------------------
# STEP 2: Knowledge Sources
# -------------------------
urls = {
    # Add URLs here if you want to include online PDFs or webpages
}

local_files = [
    #add your knowlegde base as text files 
]

chunks = []
sources = []
chunk_size = 500

# Process online sources (if any)
for condition, url in urls.items():
    print(f"Processing {condition}...")
    if url.endswith(".pdf"):
        extracted = extract_pdf_text_from_url(url)
    else:
        html = trafilatura.fetch_url(url)
        extracted = trafilatura.extract(html) if html else None
    if not extracted:
        print(f"No text extracted from {url}")
        continue
    split_chunks = [extracted[i:i+chunk_size] for i in range(0, len(extracted), chunk_size)]
    for chunk in split_chunks:
        chunks.append(f"Condition: {condition}\n{chunk.strip()}")
        sources.append(condition)

# Process local files
for file_path in local_files:
    print(f"Processing local file: {file_path}")
    extracted = extract_text_from_local_file(file_path)
    if not extracted:
        print(f"Could not extract text from {file_path}")
        continue
    condition_label = f"Manual: {file_path.split('/')[-1]}"
    split_chunks = [extracted[i:i+chunk_size] for i in range(0, len(extracted), chunk_size)]
    for chunk in split_chunks:
        chunks.append(f"Condition: {condition_label}\n{chunk.strip()}")
        sources.append(condition_label)

print(f"\nTotal chunks created: {len(chunks)}")


# -------------------------
# STEP 3: FAISS Index
# -------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Creating embeddings for chunks... this may take a moment.")
embeddings = embedding_model.encode(chunks)
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("FAISS index created.")

# -------------------------
# STEP 4: OpenAI API Chat Completion
# -------------------------
def openai_chat_completion(prompt, model="gpt-3.5-turbo"):
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
        n=1,
    )
    return response.choices[0].message.content.strip()


# -------------------------
# STEP 5: Conversation Functions
# -------------------------
conversation_history = []

def retrieve_context(query, k=4):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

def autocorrect_query(query):
    return str(TextBlob(query).correct())

def build_conversational_prompt(history, current_question, context):
    past_turns = "\n".join([f"User: {u}\nBot: {b}" for u, b in history])
    prompt = (
        "You are a knowledgeable and friendly healthcare assistant. Answer the user's question precisely using the context below.\n"
        "If the answer to question can be can be confidently inferred from context provide it with a caution 'Inferred from context'.\n\n"
        "Be precise and answer accurately as patients are usually in hurry and wrong answer can be fatal.\n\n"
        f"Context:\n{context}\n\n"
        f"{past_turns}\nUser: {current_question}\nBot:"
    )
    return prompt

def extract_last_condition(history):
    for user, bot in reversed(history):
        match = re.search(r"Condition: (.+?)\n", bot)
        if match:
            return match.group(1)
    return None

# -------------------------
# STEP 6: Chat Loop
# -------------------------
print("\nYou can start chatting with the bot now! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    last_condition = extract_last_condition(conversation_history)
    if last_condition and len(user_input.split()) <= 7:
        user_input = f"For someone with {last_condition}, {user_input}"

    context_chunks = retrieve_context(user_input)
    context = "\n".join(context_chunks)
    #print("#############################################################################\n")
    #print("\n--- Retrieved Context Preview ---")
    #print(context[:500])  # show first 500 chars of context for debug
    #print("#############################################################################\n")
    prompt = build_conversational_prompt(conversation_history, user_input, context)

    response = openai_chat_completion(prompt, model="gpt-4")

    conversation_history.append((user_input, response))
    #print("\n--- Prompt Sent to Model Preview ---\n", prompt[:1000])
    print("##############################################################################\n")
    print("Bot:", response)
    print("##############################################################################\n")
