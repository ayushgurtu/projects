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

    def generate_mcq_abstract(self) -> MCQQuestion:
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

        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert in designing abstract reasoning questions for cognitive assessments. Generate a high-quality abstract reasoning question that tests pattern recognition, logical sequences, or abstract relationships.

            Guidelines:
            - Use only characters and symbols that can be typed on a standard computer keyboard (ASCII).
            - Do not use visual shapes, diagrams, or images.
            - Avoid cultural or language-based clues.
            - The question can vary in difficulty (from medium to hard).
            - Provide **exactly 5 multiple-choice options**.
            - Include the correct answer and a brief explanation of the logic or pattern.
            - Format it in plain text (ASCII) or describe it clearly if visual.
            - Review the previous conversation history and ensure that the generated question is not a duplicate or close paraphrase of any previously generated question.
            - Output the response in valid **JSON format** with the following structure:

            {{
                "question": "<Describe the question clearly>",
                "options": ["<Option A>", "<Option B>", "<Option C>", "<Option D>", "<Option E>"],
                "correct_answer": "<Correct option>",
                "explanation": "<Explanation of the pattern or logic>"
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
                return json.loads(response)
                # parsed_response = mcq_parser.parse(response)
                
                # Validate the generated question meets requirements
                # if not parsed_response.question or len(parsed_response.options) != 5 or not parsed_response.correct_answer:
                #     raise ValueError("Invalid question format")
                # if parsed_response.correct_answer not in parsed_response.options:
                #     raise ValueError("Correct answer not in options")
                
                # return parsed_response
            except Exception as e:
                # On final attempt, raise error; otherwise continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts: {str(e)}")
                continue
    

    def generate_mcq_deductive(self) -> MCQQuestion:
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

        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a psychometric test designer. Generate a verbal deductive reasoning question suitable for an aptitude test. The question must present a short passage or set of statements from which a conclusion must be logically derived.

            Return the output in the following JSON format:

            {{
                "question": "<Describe the verbal reasoning scenario or statements>",
                "options": ["<Option A>", "<Option B>", "<Option C>", "<Option D>", "<Option E>"],
                "correct_answer": "<Correct option>",
                "explanation": "<Explanation of the deductive logic used to derive the correct answer>"
            }}

            Guidelines:
            - The question should rely purely on deductive logic from the given text, not external knowledge.
            - Only one correct answer among the five.
            - The explanation must clearly walk through the reasoning used.
            - The language should be clear and neutral, suitable for professional aptitude tests.
            - Review the previous conversation history and ensure that the generated question is not a duplicate or close paraphrase of any previously generated question.

            Generate ONE such verbal deductive reasoning question.

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
                return json.loads(response)
                # parsed_response = mcq_parser.parse(response)
                
                # Validate the generated question meets requirements
                # if not parsed_response.question or len(parsed_response.options) != 5 or not parsed_response.correct_answer:
                #     raise ValueError("Invalid question format")
                # if parsed_response.correct_answer not in parsed_response.options:
                #     raise ValueError("Correct answer not in options")
                
                # return parsed_response
            except Exception as e:
                # On final attempt, raise error; otherwise continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts: {str(e)}")
                continue
    
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
    
