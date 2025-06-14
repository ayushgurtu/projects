# Import required libraries
import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field, validator
import json

# Load environment variables from .env file
load_dotenv()

# Define data model for Multiple Choice Questions using Pydantic
class MCQQuestion(BaseModel):
    # Define the structure of an MCQ with field descriptions
    question: str = Field(description="The question text")
    options: List[str] = Field(description="List of 5 possible answers")
    # correct_answer: str = Field(description="The correct answer from the options")

    # Custom validator to clean question text
    # Handles cases where question might be a dictionary or other format
    @validator('question', pre=True)
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))
        return str(v)

class QuestionGenerator:
    def __init__(self, model):
        """
        Initialize question generator with Groq API
        Sets up the language model with specific parameters:
        - Uses llama-3.1-8b-instant model
        - Sets temperature to 0.9 for creative variety
        """
        self.llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'), 
            model=model,
            temperature=0.9
        )

        # Create memory to maintain chat history
        self.memory = ConversationBufferMemory(return_messages=True)

        self.Personality_Traits = ['Conscientiousness', 'Extraversion', 'Agreeableness', 'Emotional Stability', 'Openness to Experience']
        self.Workplace_Behaviors = ['Teamwork', 'Problem-solving', 'Adaptability', 'Initiative', 'Communication', 'Time Management']

    def generate_mcq(self, topic: str) -> MCQQuestion:
        """
        Generate Multiple Choice Question with robust error handling
        Includes:
        - Output parsing using Pydantic
        - Structured prompt template
        - Multiple retry attempts on failure
        - Validation of generated questions
        """
        # Set up Pydantic parser for type checking and validation
        mcq_parser = PydanticOutputParser(pydantic_object=MCQQuestion)

        if topic == 'Personality Traits':
            # Define the prompt template with specific format requirements
            topic = ', '.join(self.Personality_Traits)
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """
                    Generate a question designed to assess one of the following topics in a professional context: 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Emotional Stability', 'Openness to Experience'.
                    Each question should be answered using one of the following options:
                    "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree".
                 
                    Review the previous conversation history and ensure that the generated question is not a duplicate or close paraphrase of any previously generated question.

                    You must return a valid JSON object with the following structure:
                    {{
                    "question": "A clear, specific question",
                    "options": ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]
                    }}

                    Return ONLY the JSON. Do not add explanations, formatting, or markdown. """),
                    MessagesPlaceholder(variable_name="history"),  # ✅ This is how memory is injected
                    ("human", "{input}")  # Insert user input dynamically
            ])
        else:
            topic = ', '.join(self.Workplace_Behaviors)
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """
                Generate a behavioral question that assesses a candidate's general tendencies towards one of the following topics in the workplace: 'Teamwork', 'Problem-solving', 'Adaptability', 'Initiative', 'Communication', 'Time Management'.
                Each question should be answered using one of the following options:
                "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree".

                Review the previous conversation history and ensure that the generated question is not a duplicate or close paraphrase of any previously generated question.

                You must return a valid JSON object with the following structure:
                {{
                "question": "A clear, specific question",
                "options": ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]
                }}

                Return ONLY the JSON. Do not add explanations, formatting, or markdown.
            """),
            MessagesPlaceholder(variable_name="history"),  # Inject memory here
            ("human", "{input}")  # Insert user input dynamically
        ])
        # Generate response using LLM
        # Set up the chain using the Groq model, memory, and custom prompt template
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            memory=self.memory,
            verbose=False
        )
        # Implement retry logic with maximum attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Generate response using LLM
                response = chain.run(input= " ")
                parsed_response = mcq_parser.parse(response)
                
                # Validate the generated question meets requirements
                if not parsed_response.question or len(parsed_response.options) != 5:
                    raise ValueError("Invalid question format")
                
                return parsed_response
            except Exception as e:
                # On final attempt, raise error; otherwise continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts: {str(e)}")
                continue

def get_response(model, result):
    llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'), 
        model=model,
        temperature=0.9
    )

    question = result['question']
    user_response = result['user_answer']

    prompt_template = ChatPromptTemplate.from_template("""
You are an intelligent psychometric analysis agent.

Given a psychometric question and a user's Likert-scale response, do the following:

1. Identify the most relevant psychological dimension the question assesses.
2. Evaluate the Likert-scale response by mapping it to a normalized score (range: 0.0 to 1.0).
3. Assign a label based on the score:
   - "Low" for 0.0–0.33,
   - "Moderate" for >0.33–0.66,
   - "High" for >0.66–1.0.
4. Write a brief reasoning describing what the user's response reveals about their behavior or personality related to the inferred dimension.

Use only the following list of dimensions:

Personality Traits:
- Conscientiousness
- Extraversion
- Agreeableness
- Emotional Stability
- Openness to Experience

Workplace Behaviors:
- Teamwork
- Problem-solving
- Adaptability
- Initiative
- Communication
- Time Management

Input:
Question: {question}  
Likert Response: {user_response} (e.g., "Strongly Agree", "Disagree", etc.)

⛔ Important:
- You must return **only** a valid JSON object.
- Do **not** include any explanation, headers, bullet points, or markdown.
- The JSON must be the only thing in your output.

Output format:
{{
  "inferred_dimension": "<inferred dimension from list>",
  "original_response": "{user_response}",
  "normalized_score": float between 0.0 and 1.0,
  "label": "Low" | "Moderate" | "High",
  "reasoning": "What the user's response implies about their behavior or personality in relation to the inferred dimension."
}}

Example Input:
Question: "I often take the lead when a new task is assigned to my team."  
Likert Response: "Strongly Agree"

Expected Output:
{{
  "inferred_dimension": "Initiative",
  "original_response": "Strongly Agree",
  "normalized_score": 1.0,
  "label": "High",
  "reasoning": "The user consistently takes charge in team settings, indicating a strong sense of initiative and leadership."
}}
""")

    # Generate response using LLM
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=False
    )

    response = chain.run({
        "question": question,
        "user_response": user_response
    }).strip()

    # Implement retry logic with maximum attempts
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response_dict = json.loads(response)
            return response_dict
        except json.JSONDecodeError as e:
            if attempt == max_attempts - 1:
                print("JSON parsing failed:", e)
                print("Raw LLM output:", response)
                raise RuntimeError(f"Failed to parse LLM response as JSON after {max_attempts} attempts: {str(e)}")
            continue
            