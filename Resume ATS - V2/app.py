import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import os
import json
from dotenv import load_dotenv
from utils import get_groq_response, extract_pdf_text, prepare_prompt, generate_cover_letter, generate_updated_resume
import pandas as pd
import plotly.express as px

def init_session_state():
    """Initialize session state variables."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    if "resume_analyzed" not in st.session_state:
        st.session_state.resume_analyzed = False

    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""

    if "response_json" not in st.session_state:
        st.session_state.response_json = {}
    
    if "model" not in st.session_state:
        st.session_state.model = "qwen-qwq-32b"


def main():
    # Load environment variables
    load_dotenv()
    
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
        return

    # Sidebar
    with st.sidebar:
        st.title("🎯 Smart ATS")
        st.subheader("About")
        st.write("""
        This smart ATS helps you:
        - Evaluate resume-job description match
        - Identify missing keywords
        - Get personalized improvement suggestions
        """)

        option = st.selectbox(
                "Model:",
                dict.keys(),
            )
        
        if dict[option] != st.session_state.model:
             st.session_state.model = dict[option]
             st.session_state.processing = False
             st.session_state.resume_analyzed = False
             st.session_state.resume_text = ""
             st.session_state.response_json = {}

    # Main content
    st.title("📄 Smart ATS Resume Analyzer")
    st.subheader("Optimize Your Resume for ATS")
    
    # Input sections with validation
    jd = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...",
        help="Enter the complete job description for accurate analysis"
    )
    
    uploaded_file = st.file_uploader(
        "Resume (PDF)",
        type="pdf",
        help="Upload your resume in PDF format"
    )

    # Process button with loading state
    if st.button("Analyze Resume", disabled=st.session_state.processing):
        if not jd:
            st.warning("Please provide a job description.")
            return
            
        if not uploaded_file:
            st.warning("Please upload a resume in PDF format.")
            return
            
        st.session_state.processing = True
        
        try:
            with st.spinner("📊 Analyzing your resume..."):
                # Extract text from PDF
                resume_text = extract_pdf_text(uploaded_file)
                
                # Prepare prompt
                input_prompt = prepare_prompt(resume_text, jd)
                
                # Get and parse response
                response = get_groq_response(dict[option],api_key, input_prompt)
                response_json = json.loads(response)

                # Save to session_state
                st.session_state.resume_text = resume_text
                st.session_state.response_json = response_json
                st.session_state.resume_analyzed = True
                
                # Display results
                st.success("✨ Analysis Complete!")
                
                # # Match percentage
                # match_percentage = response_json.get("JD Match", "N/A")
                # st.metric("Match Score", match_percentage)
                
                # # Missing keywords
                # st.subheader("Missing Keywords")
                # missing_keywords = response_json.get("MissingKeywords", [])
                # if missing_keywords:
                #     st.write(", ".join(missing_keywords))
                # else:
                #     st.write("No critical missing keywords found!")
                
                # # Profile summary
                # st.subheader("Profile Summary")
                # st.write(response_json.get("Profile Summary", "No summary available"))
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            st.session_state.processing = False
            st.session_state.active_tab = "Skills Analysis 📊"
    

    if st.session_state.resume_analyzed: 
            # Match Overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Overall Match 🎯",
                        f"{st.session_state.response_json.get('overall_match_percentage', '0%')}"
                    )
                with col2:
                    st.metric(
                        "Skills Match 🧠",
                        f"{len(st.session_state.response_json.get('matching_skills', []))} skills"
                    )
                with col3:
                    st.metric(
                        "Skills to Develop 📈",
                        f"{len(st.session_state.response_json.get('missing_skills', []))} skills"
                    )     

                # Detailed Analysis Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Skills Analysis 📊",
                    "Experience Match 🗂️",
                    "Recommendations 💡",
                    "Cover Letter 💌",
                    "Updated Resume 📝"
                ])

                with tab1:
                    st.subheader("Matching Skills")
                    for skill in st.session_state.response_json.get('matching_skills', []):
                        st.success(f"✅ {skill['skill_name']}")

                    st.subheader("Missing Skills")
                    for skill in st.session_state.response_json.get('missing_skills', []):
                        st.warning(f"⚠️ {skill['skill_name']}")
                        st.info(f"Suggestion: {skill['suggestion']}")

                    # Skills analysis graph
                    matching_skills_count = len(st.session_state.response_json.get('matching_skills', []))
                    missing_skills_count = len(st.session_state.response_json.get('missing_skills', []))

                    skills_data = pd.DataFrame({
                        'Status': ['Matching', 'Missing'],
                        'Count': [matching_skills_count, missing_skills_count]
                    })

                    fig = px.bar(skills_data, x='Status', y='Count', color='Status',
                                 color_discrete_sequence=['#5cb85c', '#d9534f'],
                                 title='Skills Analysis')
                    fig.update_layout(xaxis_title='Status', yaxis_title='Count')

                    st.plotly_chart(fig)

                with tab2:
                    st.write("### Experience Match Analysis 🗂️")
                    st.write(st.session_state.response_json.get('experience_match_analysis', ''))
                    st.write("### Education Match Analysis 🎓")
                    st.write(st.session_state.response_json.get('education_match_analysis', ''))

                with tab3:
                    st.write("### Key Recommendations 🔑")
                    for rec in st.session_state.response_json.get('recommendations_for_improvement', []):
                        st.info(f"**{rec['recommendation']}**")
                        st.write(f"**Section:** {rec['section']}")
                        st.write(f"**Guidance:** {rec['guidance']}")

                    st.write("### ATS Optimization Suggestions 🤖")
                    for suggestion in st.session_state.response_json.get('ats_optimization_suggestions', []):
                        st.write("---")
                        st.warning(f"**Section to Modify:** {suggestion['section']}")
                        if suggestion.get('current_content'):
                            st.write(f"**Current Content:** {suggestion['current_content']}")
                        st.write(f"**Suggested Change:** {suggestion['suggested_change']}")
                        if suggestion.get('keywords_to_add'):
                            st.write(f"**Keywords to Add:** {', '.join(suggestion['keywords_to_add'])}")
                        if suggestion.get('formatting_suggestion'):
                            st.write(f"**Formatting Changes:** {suggestion['formatting_suggestion']}")
                        if suggestion.get('reason'):
                            st.info(f"**Reason for Change:** {suggestion['reason']}")

                
                with tab4:
                    st.write("### Cover Letter Generator 🖊️")
                    tone = st.selectbox("Select tone 🎭",
                                      ["Professional 👔", "Enthusiastic 😃", "Confident 😎", "Friendly 👋"])

                    if st.button("Generate Cover Letter ✍️"):
                        with st.spinner("✍️ Crafting your cover letter..."):
                            cover_letter = generate_cover_letter(
                                dict[option], api_key, jd, st.session_state.resume_text, st.session_state.response_json, tone.lower().split()[0])
                            st.markdown("### Your Custom Cover Letter 💌")
                            st.text_area("", cover_letter, height=400)
                            st.download_button(
                                "Download Cover Letter 📥",
                                cover_letter,
                                "cover_letter.txt",
                                "text/plain"
                            )

                with tab5:
                    st.write("### Updated Resume 📝")
                    if st.button("Generate Updated Resume ✍️"):
                        with st.spinner("✍️ Updating your resume..."):
                            updated_resume = generate_updated_resume(dict[option], api_key, jd, st.session_state.resume_text, st.session_state.response_json)
                            st.markdown("### Your Updated Resume 📝")
                            st.text_area("", updated_resume, height=400)
                            # Provide a download button for the updated resume
                            st.download_button(
                                "Download Updated Resume 📥",
                                updated_resume,
                                "updated_resume.pdf",
                                mime="application/pdf"
                            )

if __name__ == "__main__":
    main()
