import PyPDF2 as pdf
import json
from groq import Groq
import re

def extract_pdf_text(uploaded_file):
    """Extract text from PDF with enhanced error handling."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        if len(reader.pages) == 0:
            raise Exception("PDF file is empty")
            
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
                
        if not text:
            raise Exception("No text could be extracted from the PDF")
            
        return " ".join(text)
        
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def prepare_prompt(resume_text, job_description):
    """Prepare the input prompt with improved structure and validation."""
    if not resume_text or not job_description:
        raise ValueError("Resume text and job description cannot be empty")
        
    prompt_template = """
    Act as an expert ATS (Applicant Tracking System) specialist with deep expertise in:
    - Technical fields
    - Software engineering
    - Data science
    - Data analysis
    - Big data engineering
    
    Evaluate the following resume against the job description. Consider that the job market 
    is highly competitive. Provide detailed feedback for resume improvement.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Provide a response in the following JSON format ONLY:
    {{
"overall_match_percentage":"85%",
"matching_skills":[{{"skill_name":"Python","is_match":true}},{{"skill_name":"AWS","is_match":true}}],
"missing_skills":[{{"skill_name":"Docker","is_match":false,"suggestion":"Consider obtaining Docker certification"}}],
"skills_gap_analysis":{{"technical_skills":"Specific technical gap analysis","soft_skills":"Specific soft skills gap analysis"}},
"experience_match_analysis":"Detailed experience match analysis",
"education_match_analysis":"Detailed education match analysis",
"recommendations_for_improvement":[{{"recommendation":"Add metrics","section":"Experience","guidance":"Quantify achievements with specific numbers"}}],
"ats_optimization_suggestions":[{{"section":"Skills","current_content":"Current format","suggested_change":"Specific change needed","keywords_to_add":["keyword1","keyword2"],"formatting_suggestion":"Specific format change","reason":"Detailed reason"}}],
"key_strengths":"Specific key strengths",
"areas_of_improvement":"Specific areas to improve"
}}
Focus on providing detailed, actionable insights for each field. Keep the JSON structure exact but replace the example content with detailed analysis based on the provided job and resume.

    """
    
    return prompt_template.format(
        resume_text=resume_text.strip(),
        job_description=job_description.strip()
    )

def get_groq_response(model ,api_key ,prompt):
    """Generate a response using Gemini with enhanced error handling and response validation."""
    try:
        client = Groq(
        api_key=api_key,
        )
        
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
                    response_format={"type": "json_object"},
                    stop=None,
                )
        
        response = completion.choices[0].message.content
        
        # Ensure response is not empty
        if not response:
            raise Exception("Empty response received from Groq")
            
        # Try to parse the response as JSON
        try:
            response_json = json.loads(response)
            
            # Validate required fields
            required_fields = ["overall_match_percentage", "matching_skills", "missing_skills", "skills_gap_analysis", "experience_match_analysis", "education_match_analysis", "recommendations_for_improvement", "ats_optimization_suggestions", "key_strengths", "areas_of_improvement"]
            for field in required_fields:
                if field not in response_json:
                    raise ValueError(f"Missing required field: {field}")
                    
            return response
            
        except json.JSONDecodeError:
            # If response is not valid JSON, try to extract JSON-like content
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response.text, re.DOTALL)
            if match:
                return match.group()
            else:
                raise Exception("Could not extract valid JSON response")
                
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

def dict_to_bullet_points(d: dict, indent: int = 0) -> str:
    bullet_lines = []
    prefix = "  " * indent + "- "

    for key, value in d.items():
        if isinstance(value, dict):
            bullet_lines.append(f"{prefix}{key}:")
            bullet_lines.append(dict_to_bullet_points(value, indent + 1))
        elif isinstance(value, list):
            bullet_lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    bullet_lines.append(dict_to_bullet_points(item, indent + 1))
                else:
                    bullet_lines.append(f"{'  ' * (indent + 1)}- {item}")
        else:
            bullet_lines.append(f"{prefix}{key}: {value}")

    return "\n".join(bullet_lines)



def generate_cover_letter(model, api_key, job_analysis, resume_analysis, match_analysis,
                          tone: str = "professional") -> str:
    prompt = """
    Generate a compelling cover letter using this information (NO PREAMBLE):
    Job Details:
    {job}
    Candidate Details:
    {resume}
    Match Analysis:  
    {match}
    Tone: {tone}
    Requirements:
    1. Make it personal and specific
    2. Highlight the strongest matches
    3. Address potential gaps professionally  
    4. Keep it concise but impactful
    5. Use the specified tone: {tone}
    6. Include specific examples from the resume
    7. Make it ATS-friendly
    8. Add a strong call to action
    """

    try:
        client = Groq(api_key=api_key)

        formatted_prompt = prompt.format(
            job=job_analysis,                  
            resume=resume_analysis,            
            match=dict_to_bullet_points(match_analysis),
            tone=tone
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": " "}
            ],
            temperature=0.7,
            top_p=1,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content
        return (re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL))

    except Exception as e:
        return Exception(f"Error generating cover letter: {str(e)}")

def generate_updated_resume(model, api_key, job_analysis, resume_text, match_analysis):
    prompt = """
    You are an expert resume writer and career coach. Below is a candidate's current resume and the corresponding ATS match analysis against a specific job posting. 
    Your task is to revise the resume to increase its alignment with the job requirements, improve keyword optimization, and make it more ATS-friendly — without fabricating 
    any experience. (NO PREAMBLE):

    Resume (Original):
    {resume}
    ATS Match Analysis:  
    {match}
    Job Description:
    {job}
    Rewrite Goals:
    1. Integrate missing or underrepresented keywords/skills from the analysis (authentically and accurately).
    2. Improve alignment of experience and achievements with the job description.
    3. Enhance clarity, formatting, and ATS-compatibility (e.g., avoid complex formatting, use common headers).
    4. Do not fabricate any information. Only rephrase, highlight, or reorganize existing content and add reasonable inferred details.

    Please provide the improved resume, in plain text format
    """

    try:
        client = Groq(api_key=api_key)

        formatted_prompt = prompt.format(
            job=job_analysis,                  
            resume=resume_text,            
            match=dict_to_bullet_points(match_analysis),
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": " "}
            ],
            temperature=0.7,
            top_p=1,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content
        return (re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL))

    except Exception as e:
        return Exception(f"Error generating cover letter: {str(e)}")