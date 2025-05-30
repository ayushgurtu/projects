import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()
# Configure Generative AI
api_key = os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def init_session_state():
    """Initialize session state variables."""
    if "model" not in st.session_state:
        st.session_state.model = 'deepseek-r1-distill-llama-70b'

# Streamlit app
st.title("Resume Ranking System")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

def rank_resumes_semantic(job_description, resumes):
    model = load_model()
    embeddings = model.encode([job_description] + resumes, convert_to_tensor=True)

    job_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]

    cosine_scores = util.cos_sim(job_embedding, resume_embeddings)
    return cosine_scores[0].cpu().numpy()  # Similarity scores as list

def score_resume_with_llm(model ,api_key, job_description, resume_text):
    """Score a resume using a Generative AI model."""
    client = Groq(
        api_key=api_key,
        )
    
    prompt = f"""
            You are a hiring assistant. Evaluate the following resume against this job description and provide a score from 0 to 100 for how well it fits. Also provide a short explanation.

            Job Description:
            {job_description}

            Resume:
            {resume_text}

            Return the output in the format:
            Score: <score>
            Explanation: <reason>
            """
        
    completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": " "
                    }
                ],
                temperature=1,
                # max_completion_tokens=8192,
                top_p=1,
                stream=False,
                stop=None,
            )
        
    response = completion.choices[0].message.content

    # Parse the response
    score = 0
    explanation = ""
    for line in response.split("\n"):
        if line.startswith("Score:"):
            score = int(line.split(":")[1].strip())
        elif line.startswith("Explanation:"):
            explanation = line.split(":")[1].strip()

    return score, explanation


# Initialize session state
init_session_state()

# Configure Generative AI
api_key = os.getenv("GROQ_API_KEY")
dict = {
    # "QwQ 32B": "qwen-qwq-32b",
    "DeepSeek R1 Distill Llama 70B": 'deepseek-r1-distill-llama-70b',
    "Gemma 2 Instruct": "gemma2-9b-it",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 4 Maverick 17B 128E": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Llama 4 Scout 17B 16E": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Mistral Saba 24B": "mistral-saba-24b"
}
if not api_key:
    st.error("Please set the GROQ_API_KEY in your .env file")

# Sidebar
with st.sidebar:
    st.header("LLM Model Configuration")
    st.write("Select the model you want to use for resume ranking:")
    option = st.selectbox(
            "Model:",
            dict.keys(),
        )
    
    if dict[option] != st.session_state.model:
            st.session_state.model = dict[option]


# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")

    tab1, tab2, tab3 = st.tabs([
            "TF-IDF",
            "Sentence Transformers",
            "LLM-Based"
        ])

    with tab1:
        resumes = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)
            
        # Rank resumes based on TF-IDF
        scores = rank_resumes(job_description, resumes)
        
        # Display scores
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False, ignore_index=True)
        
        st.write(results)

    with tab2:
        resumes = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)
        # Rank resumes based on Sentence Transformers
        scores = rank_resumes_semantic(job_description, resumes)

        # Display scores
        results1 = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results1 = results1.sort_values(by="Score", ascending=False, ignore_index=True)
        st.write(results1)
    
    with tab3:
        resumes = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)
        
        # Score each resume using the LLM
        scores = []
        explanations = []
        for resume in resumes:
            score, explanation = score_resume_with_llm(st.session_state.model, api_key, job_description, resume)
            scores.append(score)
            explanations.append(explanation)

        # Display scores and explanations
        results3 = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Score": scores,
            "Explanation": explanations
        })
        results3 = results3.sort_values(by="Score", ascending=False, ignore_index=True)
        st.write(results3)
