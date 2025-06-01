import streamlit as st
import tempfile
import json
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

st.set_page_config(
    page_title="AskDOC - AI Academic Assistant", 
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Theme configuration
def apply_theme(theme):
    if theme == "dark":
        st.markdown("""
            <style>
            :root {
                --primary-color: #6c5ce7;
                --secondary-color: #a29bfe;
                --dark-bg: #0f0f13;
                --darker-bg: #0a0a0d;
                --light-text: #f8f9fa;
                --lighter-text: #e9ecef;
                --accent-color: #fd79a8;
                --border-color: #2d3436;
                --success-color: #00b894;
                --warning-color: #fdcb6e;
                --info-color: #0984e3;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            :root {
                --primary-color: #6c5ce7;
                --secondary-color: #a29bfe;
                --dark-bg: #ffffff;
                --darker-bg: #f8f9fa;
                --light-text: #212529;
                --lighter-text: #495057;
                --accent-color: #fd79a8;
                --border-color: #dee2e6;
                --success-color: #28a745;
                --warning-color: #ffc107;
                --info-color: #17a2b8;
            }
            </style>
            """, unsafe_allow_html=True)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"

# Apply the selected theme
apply_theme(st.session_state.theme)

# Combined CSS styles with all fixes
st.markdown(f"""
    <style>
    /* ===== BASE STYLES ===== */
    .main {{
        background-color: var(--dark-bg);
        color: var(--light-text);
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: var(--light-text);
    }}
    
    .stMarkdown p {{
        color: var(--lighter-text);
    }}
            
    :root {{
        --light-mode-fix-dark-text: #212529; /* standard dark text for light mode */
    }}

    /* Apply fix ONLY in light mode (i.e. when background is light) */
    body:has(.main[style*="background-color: var(--light-bg);"]) .meta-value,
    body:has(.main[style*="background-color: var(--light-bg);"]) .document-name,
    body:has(.main[style*="background-color: var(--light-bg);"]) .answer-meta {{
        color: var(--light-mode-fix-dark-text) !important;
    }}
            
        /* ===== ASK DOC BANNER ===== */
    .ask-doc-banner {{
        background: linear-gradient(145deg, var(--darker-bg), var(--dark-bg)) !important;
        padding: 20px 30px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        margin: 35px auto;
        border-left: 8px solid var(--primary-color);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
        width: 300%;
        box-sizing: border-box;
    }}
    
    .ask-doc-title, .ask-doc-tagline {{
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5;
        line-height: 1.5;
    }}
    
    .ask-doc-title {{
        font-size: 2.0rem;
        font-weight: 700;
    }}
    
    .ask-doc-tagline {{
        font-size: 1.7rem;
        font-weight: 400;
        opacity: 0.9;
    }}
    
    .ask-doc-banner:hover {{
        transform: translateY(-3px);
        transition: transform 0.4s ease;
    }}
    /* ===== SIDEBAR STYLES ===== */
    [data-testid="stSidebar"] {{
        width: 300px !important;
        min-width: 300px !important;
        max-width: 400px !important;
    }}
    
    [data-testid="stSidebar"][aria-expanded="false"] {{
        margin-left: -300px;
        opacity: 0;
        transition: margin-left 300ms ease, opacity 300ms ease;
    }}
    
    /* Adjust main content when sidebar is collapsed */
    [data-testid="stSidebar"][aria-expanded="false"] + .main {{
        margin-left: 0;
        width: 100%;
    }}
    
    .main .block-container {{
        max-width: 90%;
        margin: auto;
    }}
    
    /* ===== COMPONENT STYLES ===== */
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    }}
    
    .stButton>button:active {{
        transform: translateY(0);
    }}
    
    /* Input Fields */
    .stTextInput>div>div>input {{
        border-radius: 12px;
        padding: 12px;
        background-color: var(--darker-bg);
        color: var(--light-text);
        border: 2px solid var(--border-color);
        transition: all 0.3s;
    }}
    
    .stTextInput>div>div>input:focus {{
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(108, 92, 231, 0.2);
    }}
    
    /* Selectbox - Custom Styled */
    .stSelectbox > div[data-baseweb="select"] {{
        width: 250px;
        border-radius: 12px;
        font-size: 16px;
        padding: 8px 12px;
        background-color: var(--darker-bg);
        color: var(--light-text);
        border: 2px solid var(--border-color);
        transition: all 0.3s;
    }}
    
    .stSelectbox > div[data-baseweb="select"]:hover {{
        border-color: var(--primary-color);
        box-shadow: 0 0 0 0.2rem rgba(108, 92, 231, 0.25);
    }}
    
    /* File Uploader */
    .stFileUploader {{
        border-radius: 12px;
        padding: 25px;
        background: var(--darker-bg);
        border: 2px dashed var(--border-color);
        transition: all 0.3s;
        color : #6c5ce7;
    }}
    
    .stFileUploader:hover {{
        border-color: var(--primary-color);
    }}
    
    /* Text Areas */
    .stTextArea textarea {{
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 2px solid var(--primary-color);
        border-radius: 12px;
        padding: 12px;
    }}
    
    /* Expanders */
    .st-expander {{
        background-color: var(--darker-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .st-expanderHeader {{
        color: var(--light-text);
        font-weight: 700;
        font-size: 1.1rem;
    }}
    
    /* ===== LAYOUT COMPONENTS ===== */
    .header {{
        color: var(--light-text);
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 15px;
        margin-bottom: 25px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    
    .footer {{
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        color: var(--lighter-text);
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
    }}
    
    /* ===== ASK DOC BANNER ===== */
    .ask-doc-banner {{
        background: color-mix(in srgb, var(--dark-bg) 5%, white 95%) !important;
        color: blue;
        padding: 14px 24px;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        margin: 25px auto;
        border-left: 6px solid var(--primary-color);
        display: block;
        width: fit-content;
        max-width: 90%;
        box-sizing: border-box;
    }}
    
    .divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 25px 0;
    }}
    
    /* ===== ANSWER SECTION ===== */
    .answer-section {{
        background: linear-gradient(145deg, var(--darker-bg), var(--dark-bg));
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
        border-left: 5px solid var(--primary-color);
        color: var(--light-text);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }}
    
    .answer-header {{
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--light-text);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .answer-content {{
        font-size: 1.15rem;
        line-height: 1.7;
        color: var(--lighter-text);
        padding: 20px;
        background-color: var(--darker-bg);
        border-radius: 12px;
        margin: 20px 0;
    }}
    
    /* FIX FOR SOURCE ANALYSIS VISIBILITY */
    .answer-meta {{
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid var(--border-color);
        font-size: 0.95rem;
        color: #333333;
    }}
    
    .meta-item {{
        display: flex;
        margin-bottom: 8px;
        align-items: center;
    }}
    
    .meta-label {{
        font-weight: 600;
        min-width: 120px;
        color: var(--secondary-color);
    }}
    
    .meta-value {{
        font-weight: 500;
        color: var(--light-text) !important; /* Ensure visibility */
    }}
    
    /* DOCUMENT NAME VISIBILITY FIX */
    .document-name {{
        color: var(--light-text) !important;
        font-weight: 600;
    }}
    
    body.light-mode .document-name {{
        color: var(--lighter-text) !important;
    }}
    
    /* ===== TOOL OUTPUTS ===== */
    .tool-output {{
        background: linear-gradient(145deg, var(--darker-bg), var(--dark-bg));
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
        border-left: 5px solid var(--info-color);
        color: var(--light-text);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }}
    
    .tool-card {{
        background: linear-gradient(145deg, var(--darker-bg), var(--dark-bg));
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 3px solid var(--primary-color);
    }}
    
    /* ===== QUESTION INPUT ===== */
    .question-input-container {{
        background: linear-gradient(145deg, var(--darker-bg), var(--dark-bg));
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid var(--primary-color);
    }}
    
    .question-input-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
    }}
    
    .question-input-header h3 {{
        margin: 0;
        color: var(--light-text);
    }}
    
    .question-input-icon {{
        font-size: 1.5rem;
        color: var(--primary-color);
    }}
    
    /* ===== THEME SWITCHER ===== */
    .theme-switch {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
    }}
    
    .theme-switch-label {{
        font-weight: 600;
        color: var(--light-text);
    }}
    
    /* ===== SETTINGS SECTION ===== */
    .settings-section {{
        background: linear-gradient(145deg, var(--darker-bg), var(--dark-bg));
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    /* ===== PROGRESS BARS ===== */
    .progress-container {{
        width: 100%;
        background-color: var(--darker-bg);
        border-radius: 10px;
        margin: 15px 0;
        overflow: hidden;
    }}
    
    .progress-bar {{
        height: 20px;
        border-radius: 10px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
        transition: width 0.5s ease-in-out;
    }}
    
    .progress-labels {{
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 0.85rem;
        color: var(--lighter-text);
    }}
    
    /* ===== HIGHLIGHTS ===== */
    .source-highlight {{
        background-color: #FFF9C4;
        padding: 1px 1px;
        border-radius: 2px;
        border-left: 1px solid #FBC02D;
        color: #212121;
    }}
            
    .external-highlight {{
        color : #4f83cc;
    }}
    
    /* ===== CONFIDENCE INDICATORS ===== */
    .confidence-high {{
        color: var(--success-color);
        font-weight: 700;
    }}
    
    .confidence-medium {{
        color: var(--warning-color);
        font-weight: 700;
    }}
    
    .confidence-low {{
        color: var(--accent-color);
        font-weight: 700;
    }}
    
    /* ===== MOBILE RESPONSIVENESS ===== */
    @media (max-width: 768px) {{
        .stButton>button {{
            padding: 8px 16px;
            font-size: 0.9rem;
        }}
        
        .answer-content {{
            font-size: 1rem;
            padding: 15px;
        }}
        
        .stSelectbox > div[data-baseweb="select"] {{
            width: 100%;
            font-size: 14px;
        }}
        
        .ask-doc-banner {{
            font-size: 1.4rem;
            padding: 10px 15px;
        }}
        
        /* Adjust main content for mobile */
        .main .block-container {{
            max-width: 95%;
            padding: 1rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

# -------------------- Title Banner --------------------
st.markdown("""
    <div class="ask-doc-banner">
        <h1 class="ask-doc-title">AskDOC</h1>
        <p class="ask-doc-tagline">Document intelligence at your fingertips</p>
    </div>
    <p style="font-size:1.1rem; color:var(--lighter-text); margin: 0 auto 30px; max-width: 800px; text-align: center;">
    Upload your academic documents and get AI-powered answers with precise source analysis. 
    Know exactly what comes from your materials versus general knowledge.
    </p>
""", unsafe_allow_html=True)

# Initialize session state
if 'context_text' not in st.session_state:
    st.session_state.context_text = ""
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'summary_output' not in st.session_state:
    st.session_state.summary_output = ""
if 'mcqs_output' not in st.session_state:
    st.session_state.mcqs_output = ""
if 'topic_explanation_output' not in st.session_state:
    st.session_state.topic_explanation_output = ""
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'topic_query' not in st.session_state:
    st.session_state.topic_query = ""
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.3
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 4096
if 'source_percentage' not in st.session_state:
    st.session_state.source_percentage = 0
if 'highlighted_answer' not in st.session_state:
    st.session_state.highlighted_answer = ""

# -------------------- Helper Functions --------------------
def document_loader(file_path, file_name):
    if file_name.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_name.endswith(".docx"):
        return UnstructuredWordDocumentLoader(file_path).load()
    elif file_name.endswith(".txt"):
        return TextLoader(file_path).load()

def load_and_split(file_path, file_name):
    docs = document_loader(file_path, file_name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def get_confidence_class(score):
    try:
        conf = float(score)
        if conf >= 0.8:
            return "confidence-high"
        elif conf >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"
    except:
        return ""

def process_documents_for_tools(uploaded_files):
    """Process documents and return full text for tools"""
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            content = file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        docs = load_and_split(temp_file_path, file.name)
        all_docs.extend(docs)
    
    return "\n\n".join([doc.page_content for doc in all_docs])

def initialize_model(model_name, temperature, max_tokens):
    """Initialize the LLM model with current settings"""
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

def analyze_source_coverage(answer, context):
    """Analyze what percentage of the answer is covered by the context"""
    if not context:
        return 0, ""
    
    # Simple word-based analysis (could be enhanced with embeddings similarity)
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    
    common_words = answer_words.intersection(context_words)
    coverage = len(common_words) / len(answer_words) if len(answer_words) > 0 else 0
    
    # Create highlighted version of the answer
    highlighted = []
    for word in answer.split():
        if word.lower() in context_words:
            highlighted.append(f'<span class="source-highlight">{word}</span>')
        else:
            highlighted.append(f'<span class="external-highlight">{word}</span>')
    
    return min(1.0, max(0.0, coverage)), " ".join(highlighted)

# -------------------- Sidebar for Settings --------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration:")
    
    # Theme switcher
    st.markdown('<div class="theme-switch">', unsafe_allow_html=True)
    st.markdown('<span class="theme-switch-label">Theme:</span>', unsafe_allow_html=True)
    if st.button("🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        apply_theme(st.session_state.theme)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        
        model_options = {
            "Llama 3 (8B)": "llama3-8b-8192",
            "Llama 3 (70B)": "llama3-70b-8192",
            
        }

        selected_model = st.selectbox(
            "Select AI Model:",
            options=list(model_options.keys()),  # Show display names
            index=0 

        # Get the corresponding model key (e.g., "llama3-8b-8192")
        model_key = model_options[selected_model]
        st.write(f"Selected model key: `{model_key}`")
        
        # Temperature slider
        st.session_state.temperature = st.slider(
            "Creativity (Temperature):",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Higher values make outputs more creative but less factual"
        )
        
        # Max tokens
        st.session_state.max_tokens = st.slider(
            "Max Response Length:",
            min_value=512,
            max_value=8192,
            value=4096,
            step=512,
            help="Maximum number of tokens in the response"
        )
        
        # Initialize model with current settings
        st.session_state.llm_model = initialize_model(
            model_options[selected_model],
            st.session_state.temperature,
            st.session_state.max_tokens
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Document upload in sidebar
    st.markdown("## 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload academic documents (PDF, DOCX, or TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="font-size:0.9rem; color:var(--lighter-text);">
        <p><strong>🔍 Source Analysis</strong></p>
        <p>The assistant highlights:</p>
        <ul>
            <li><span class="source-highlight" style="padding:0 4px;">Content from your documents</span></li>
            <li><span class="external-highlight" style="padding:0 4px;">General knowledge</span></li>
        </ul>
        <p>You'll see exactly what comes from your materials.</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Main Content Area --------------------
# Question Input Section
with st.container():
    st.markdown("""
        <div class="question-input-container">
            <div class="question-input-header">
                <span class="question-input-icon">❓</span>
                <h3>Ask Your Question</h3>
            </div>
    """, unsafe_allow_html=True)
    
    question = st.text_area(
        "Enter your academic question here...", 
        placeholder="e.g., Explain quantum entanglement in simple terms...\nOr ask about specific concepts from your documents...",
        label_visibility="collapsed",
        key="question_input",
        height=100
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Process Button
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("🔍 Analyze & Answer", key="get_answer", use_container_width=True):
        if not uploaded_files or not question:
            st.warning("Please upload documents and enter a question")
        else:
            with st.spinner("Analyzing documents and generating answer..."):
                try:
                    # Process uploaded files
                    all_docs = []
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
                            content = file.read()
                            temp_file.write(content)
                            temp_file_path = temp_file.name
                        
                        docs = load_and_split(temp_file_path, file.name)
                        all_docs.extend(docs)
                    
                    # Store full text for tools
                    st.session_state.full_text = "\n\n".join([doc.page_content for doc in all_docs])
                    
                    # Create vector store
                    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(all_docs, embedding=embedding)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    results = retriever.invoke(question)
                    
                    # Build context text
                    context_text = ""
                    for doc in results:
                        if len(context_text) + len(doc.page_content) <= 5000:
                            context_text += doc.page_content + "\n\n"
                    
                    st.session_state.context_text = context_text
                    
                    # Prepare and run the chain
                    parser = StrOutputParser()
                    prompt = PromptTemplate(
                        template="""You are a precise academic assistant. Answer the question based primarily on the provided context. 
                        For any information not in the context, clearly indicate it as general knowledge. Be detailed and accurate.

                        Context:
                        {context}

                        Question: {question}

                        Structured Answer:""",
                        input_variables=["context", "question"]
                    )
                    
                    chain = RunnableSequence(
                        RunnablePassthrough()
                        | (lambda q: {"context": context_text, "question": q})
                        | prompt
                        | st.session_state.llm_model
                        | parser
                    )
                    
                    raw_output = chain.invoke(question)
                    
                    # Analyze source coverage
                    coverage_percentage, highlighted = analyze_source_coverage(raw_output, context_text)
                    st.session_state.source_percentage = coverage_percentage
                    st.session_state.highlighted_answer = highlighted
                    
                    # Create response object
                    response = {
                        "question": question,
                        "answer": raw_output,
                        "source_document": "Uploaded Documents",
                        "confidence_score": str(min(0.95, max(0.7, coverage_percentage * 0.9))),  # Scale confidence based on coverage
                        "coverage_percentage": coverage_percentage,
                        "highlighted_answer": highlighted
                    }
                    
                    st.session_state.current_response = response
                    st.success("Answer generated with source analysis!")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

with col2:
    if st.button("🔄 Clear Session", type="secondary", use_container_width=True):
        st.session_state.context_text = ""
        st.session_state.vectorstore = None
        st.session_state.summary_output = ""
        st.session_state.mcqs_output = ""
        st.session_state.topic_explanation_output = ""
        st.session_state.current_response = None
        st.session_state.full_text = ""
        st.session_state.topic_query = ""
        st.session_state.source_percentage = 0
        st.session_state.highlighted_answer = ""
        st.rerun()

if st.session_state.current_response:
    with st.container():
        st.subheader("📄 Answer Analysis")
        with st.expander("View Detailed Response", expanded=True):
            response = st.session_state.current_response

            conf_class = get_confidence_class(response.get('confidence_score', '0.85'))
            
            # Source coverage visualization
            coverage_percentage = response.get('coverage_percentage', 0)
            st.markdown(f"""
                <div style="margin-bottom:25px;">
                    <h4>Source Coverage</h4>
                    <p>This answer is derived <strong>{coverage_percentage*100:.1f}%</strong> from your documents:</p>
                    <div class="progress-container">
                        <div class="progress-bar" style="width:{coverage_percentage*100}%">
                            {coverage_percentage*100:.1f}%
                        </div>
                    </div>
                    <div class="progress-labels">
                        <span>Document Content</span>
                        <span>General Knowledge</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="tool-output">
                    <div class="answer-header">
                        <span>Question:</span>
                        <span>{response.get('question', question)}</span>
                    </div>
                    <div class="answer-content">
                        {response.get('answer', '')}
                    </div>
                    <div class="answer-meta">
                        <div class="meta-item">
                            <span class="meta-label">Confidence Score:</span>
                            <span class="meta-value {conf_class}">
                                {response.get('confidence_score', '0.85')}
                            </span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Source Coverage:</span>
                            <span class="meta-value">
                                {coverage_percentage*100:.1f}% from documents
                            </span>
                        </div>
                    
                
            """, unsafe_allow_html=True)
            
            # Display the highlighted version with a checkbox
            if st.checkbox("🔍 Show Source Analysis", key="show_source_analysis"):
                st.markdown(f"""
                    <div style="padding:15px; background-color:var(--darker-bg); border-radius:12px;">
                        <h4>Answer with Source Highlighting</h4>
                        <div style="margin-top:15px;">
                            {response.get('highlighted_answer', response.get('answer', ''))}
                        </div>
                        <div style="margin-top:15px; font-size:0.9rem; color:var(--lighter-text);">
                            <p><span class="source-highlight" style="padding:2px 6px;">Highlighted</span> content comes from your documents</p>
                            <p><span class="external-highlight" style="padding:2px 6px;">Highlighted</span> content is general knowledge</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# -------------------- Tool Functions --------------------
def generate_summary():
    """Generate summary of the content from document"""
    if not st.session_state.full_text:
        return "No document content available"
    
    prompt = PromptTemplate(
        template="""Create a comprehensive summary of the following academic content. 
        Focus on key concepts, main arguments, and important details. 
        Organize the summary with clear headings and bullet points where appropriate.
        
        Content:
        {context}
        
        Structured Summary:""",
        input_variables=["context"]
    )
    chain = (
        RunnablePassthrough()
        | (lambda _: {"context": st.session_state.full_text[:8000]})
        | prompt
        | st.session_state.llm_model
        | StrOutputParser()
    )
    return chain.invoke({})

def generate_mcqs():
    """Generate MCQs using a direct chain"""
    if not st.session_state.full_text:
        return "No document content available"
    
    prompt = PromptTemplate(
        template="""Generate 5 high-quality multiple-choice questions based on the content below. 
        For each question:
        - Provide a clear, well-formulated question
        - Include 4 plausible options (a-d)
        - Mark the correct answer with (Correct)
        - Include a brief explanation for the correct answer
        
        Format each question like this:
        ### Question 1
        [Question text]
        a) Option 1
        b) Option 2
        c) Option 3 (Correct)
        d) Option 4
        Explanation: [Brief explanation]
        
        Content:
        {context}
        
        Questions:""",
        input_variables=["context"]
    )
    chain = (
        RunnablePassthrough()
        | (lambda _: {"context": st.session_state.full_text[:8000]})
        | prompt
        | st.session_state.llm_model
        | StrOutputParser()
    )
    return chain.invoke({})

def generate_topic_explanation(topic):
    """Generate topic explanation"""
    if not st.session_state.full_text:
        return "No document content available"
    if not topic:
        return "Please enter a topic"
    
    prompt = PromptTemplate(
        template="""Provide a detailed explanation of the topic '{topic}' using the content below. 
        Structure your response with:
        1. Clear definition
        2. Key concepts
        3. Examples from the content (marked as [Source])
        4. Additional context (marked as [General Knowledge])
        
        Content:
        {context}
        
        Detailed Explanation:""",
        input_variables=["context", "topic"]
    )
    chain = (
        RunnablePassthrough()
        | (lambda _: {"context": st.session_state.full_text[:8000], "topic": topic})
        | prompt
        | st.session_state.llm_model
        | StrOutputParser()
    )
    return chain.invoke({})

def get_confidence_class(score):
    try:
        conf = float(score)
        if conf >= 0.8:
            return "confidence-high"
        elif conf >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"
    except:
        return ""
# -------------------- Bonus Tools Section --------------------
st.markdown("---")
st.markdown('<h2 class="header">🔧 Learning Tools</h2>', unsafe_allow_html=True)
st.markdown("""
    <p style="color:var(--lighter-text); margin-bottom:25px;">
    Enhance your learning experience with these academic tools. Each tool processes your uploaded documents.
    </p>
    """, unsafe_allow_html=True)

# Tools in cards
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("""
            <div class="tool-card">
                <h4><span class="icon">📝</span> Document Summary</h4>
                <p style="color:var(--lighter-text); font-size:0.95rem;">
                Generate a concise, structured summary of your uploaded documents highlighting key points.
                </p>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Generate Summary", key="summary_btn", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload documents first")
            else:
                with st.spinner("Analyzing documents and generating summary..."):
                    try:
                        if not st.session_state.full_text:
                            st.session_state.full_text = process_documents_for_tools(uploaded_files)
                        st.session_state.summary_output = generate_summary()
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                        st.session_state.summary_output = "Summary generation failed"

with col2:
    with st.container():
        st.markdown("""
            <div class="tool-card">
                <h4><span class="icon">✍️</span> Practice MCQs</h4>
                <p style="color:var(--lighter-text); font-size:0.95rem;">
                Create multiple-choice questions with explanations to test your understanding.
                </p>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Generate MCQs", key="mcqs_btn", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload documents first")
            else:
                with st.spinner("Creating practice questions from your documents..."):
                    try:
                        if not st.session_state.full_text:
                            st.session_state.full_text = process_documents_for_tools(uploaded_files)
                        st.session_state.mcqs_output = generate_mcqs()
                    except Exception as e:
                        st.error(f"Error generating MCQs: {str(e)}")
                        st.session_state.mcqs_output = "MCQ generation failed"

with col3:
    with st.container():
        st.markdown("""
            <div class="tool-card">
                <h4><span class="icon">📖</span> Topic Explanation</h4>
                <p style="color:var(--lighter-text); font-size:0.95rem;">
                Get detailed explanations of specific topics using your documents as reference.
                </p>
            </div>
            """, unsafe_allow_html=True)
        topic_query = st.text_input("Enter topic:", key="topic_input", placeholder="e.g., Quantum Mechanics", label_visibility="collapsed")
        if st.button("Explain Topic", key="topic_btn", use_container_width=True):
            if not topic_query:
                st.warning("Enter a topic first")
            elif not uploaded_files:
                st.warning("Please upload documents first")
            else:
                with st.spinner(f"Analyzing documents for '{topic_query}'..."):
                    try:
                        if not st.session_state.full_text:
                            st.session_state.full_text = process_documents_for_tools(uploaded_files)
                        st.session_state.topic_explanation_output = generate_topic_explanation(topic_query)
                        st.session_state.topic_query = topic_query
                    except Exception as e:
                        st.error(f"Error generating explanation: {str(e)}")
                        st.session_state.topic_explanation_output = "Explanation generation failed"

# -------------------- Tool Outputs --------------------
st.markdown("---")
st.subheader("✨ Generated Content")

# Summary Output
if st.session_state.summary_output:
    with st.expander("Document Summary", expanded=True):
        st.markdown(f"""
            <div class="tool-output">
                <div class="answer-header">
                    <span>Document Summary</span>
                </div>
                <div class="answer-content">
                    {st.session_state.summary_output}
                </div>
            </div>
        """, unsafe_allow_html=True)

# MCQs Output
if st.session_state.mcqs_output:
    with st.expander("Practice Questions", expanded=True):
        st.markdown(f"""
            <div class="tool-output">
                <div class="answer-header">
                    <span>Multiple Choice Questions</span>
                </div>
                <div class="answer-content">
                    {st.session_state.mcqs_output}
                </div>
            </div>
        """, unsafe_allow_html=True)

# Topic Explanation
if st.session_state.topic_explanation_output:
    with st.expander(f"Explanation of '{st.session_state.topic_query}'", expanded=True):
        st.markdown(f"""
            <div class="tool-output">
                <div class="answer-header">
                    <span>Topic Explanation: {st.session_state.topic_query}</span>
                </div>
                <div class="answer-content">
                    {st.session_state.topic_explanation_output}
                </div>
            </div>
        """, unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p style="font-size:0.8rem; margin-top:10px;">
        Powered by Groq & LangChain & Rag | Know exactly what comes from your materials
        </p>
    </div>
""", unsafe_allow_html=True)
