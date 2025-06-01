# AskDOC
AskDOC: AI-Powered Academic Research Assistant  AskDOC uses RAG &amp; Groq’s fast LLMs to analyze your PDFs, Word docs, and text files. Get precise answers with citations, summaries, MCQs, and topic explanations. Dark/light mode, adjustable creativity, and private data handling. Perfect for students &amp; researchers.
# AskDOC - AI Academic Assistant 📚

AskDOC is an advanced document intelligence tool that helps students and researchers extract insights from academic documents using AI. It combines RAG (Retrieval-Augmented Generation) with Groq's ultra-fast LLMs to provide accurate, source-aware answers.

## Features ✨
- **Document Intelligence**: Upload PDFs, Word docs, or text files and ask questions
- **Source Analysis**: Know exactly what comes from your documents vs general knowledge
- **Learning Tools**: Document summarization, MCQ generator, Topic explainer
- **Fast Inference**: Powered by Groq's LPUs for near-instant responses
- **Customizable**: Adjust creativity and response length

## Tech Stack 🛠️
- **Backend**: Python 3.9+, Streamlit, LangChain, FAISS, HuggingFace Embeddings
- **LLM Providers**: Groq (Llama 3 8B/70B, Gemma 7B)
- **Frontend**: Streamlit Components with Custom CSS

## Installation ⚙️
1. Clone the repository:
   
         git clone https://github.com/yourusername/askdoc.git
         cd askdoc

2.Set up a virtual environment:

      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.Install dependencies:

      pip install -r requirements.txt

4.Create .env file with your Groq API key:

      GROQ_API_KEY=your_api_key_here

5.Running the App 

       streamlit run app.py

How It Works 
Document Processing: Chunks documents using RecursiveCharacterTextSplitter and embeds with all-MiniLM-L6-v2

RAG Pipeline: Retrieves relevant chunks using FAISS and generates answers with Groq LLMs

Source Analysis: Highlights content origin (documents vs general knowledge)

Configuration ⚙️
Choose between Llama 3 (8B/70B) or Gemma (7B)

Adjust temperature (0.0-1.0) and max tokens (512-8192)

Toggle between light/dark modes

Requirements

    streamlit==1.31.0
    langchain==0.1.0
    langchain-community==0.0.20
    langchain-groq==0.1.0
    langchain-core==0.1.0
    faiss-cpu==1.7.4
    unstructured==0.12.0
    python-dotenv==1.0.0
    PyPDF2==3.0.1

Screenshots 📸


Main Interface  : <img width="1407" alt="Screenshot 2025-06-01 at 9 57 43 PM" src="https://github.com/user-attachments/assets/35b438c8-266d-49dc-a020-4b3782fce33f" />

Learning Tools	: <img width="1057" alt="Screenshot 2025-06-01 at 9 59 17 PM" src="https://github.com/user-attachments/assets/fb49a80a-ec64-4173-91e2-9cc2711946f6" />

Roadmap :

PowerPoint file support

Document annotation

Multi-document comparison

Citation generation



