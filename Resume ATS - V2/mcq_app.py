# Import required libraries
import streamlit as st
import pandas as pd
import random
from mcq_utils import QuestionGenerator, get_response
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# Main class to handle quiz functionality
class QuizManager:
    def __init__(self):
        # Initialize empty lists to store questions, user answers and results
        self.questions = []
        self.user_answers = []
        self.results = []

    def generate_questions(self, generator, question_type, num_questions):
        # Reset all lists before generating new questions
        self.questions = []
        self.user_answers = []
        self.results = []

        try:
            
            # Generate specified number of questions
            for _ in range(num_questions // 2):
                # Handle Multiple Choice Questions
                if question_type == "Multiple Choice":
                    question = generator.generate_mcq(topic = "Personality Traits")
                    self.questions.append({
                        'type': 'MCQ',
                        'question': question.question,
                        'options': question.options,
                        'category':  "Personality Traits"
                        # 'correct_answer': question.correct_answer
                    })
            for _ in range(num_questions // 2):
                # Handle Multiple Choice Questions
                if question_type == "Multiple Choice":
                    question = generator.generate_mcq(topic = "Workplace Behaviors")
                    self.questions.append({
                        'type': 'MCQ',
                        'question': question.question,
                        'options': question.options,
                        'category':  "Workplace Behaviors"
                        # 'correct_answer': question.correct_answer
                    })
        except Exception as e:
            # Display error if question generation fails
            st.error(f"Error generating questions: {e}")
            return False
        return True
    
    def attempt_quiz(self):
        # Display questions and collect user answers
        for i, q in enumerate(self.questions):
            # Display question with bold formatting
            st.markdown(f"**Question {i+1}: {q['question']}**")
            
            # Handle MCQ input using radio buttons
            if q['type'] == 'MCQ':
                user_answer = st.radio(
                    f"Select an answer for Question {i+1}", 
                    q['options'], 
                    key=f"mcq_{i}"
                )
                # self.user_answers.append(user_answer)
            # Update only if changed
            st.session_state.user_answers[i] = user_answer
        # Update the instance variable
        self.user_answers = st.session_state.user_answers

    def evaluate_quiz(self):
        # Reset results before evaluation
        self.results = []
        # Evaluate each question and user answer pair
        for i, (q, user_ans) in enumerate(zip(self.questions, self.user_answers)):
            # Create base result dictionary
            result_dict = {
                'question_number': i + 1,
                'question': q['question'],
                'question_type': q['type'],
                'user_answer': user_ans
            }
            response = get_response(st.session_state.model, result_dict)
            result_dict['Dimension'] = response.get('inferred_dimension', None)
            result_dict['Score'] = response.get('normalized_score', None)
            result_dict['Label'] = response.get('label', None) 
            result_dict['Reasoning'] = response.get('reasoning', None) 
            self.results.append(result_dict)

    def generate_result_dataframe(self):
        # Convert results to pandas DataFrame
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def save_to_csv(self, filename='quiz_results.csv'):
        try:
            # Check if results exist
            if not self.results:
                st.warning("No results to save. Please complete the quiz first.")
                return None
            
            # Generate DataFrame from results
            df = self.generate_result_dataframe()
            
            # Create unique filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"quiz_results_{timestamp}.csv"
            
            # Ensure results directory exists
            os.makedirs('results', exist_ok=True)
            full_path = os.path.join('results', unique_filename)
            
            # Save results to CSV
            df.to_csv(full_path, index=False)
            
            # Display success message
            st.success(f"Results saved to {full_path}")
            return full_path
        except Exception as e:
            # Handle any errors during saving
            st.error(f"Failed to save results: {e}")
            return None

def main():
    # Configure Streamlit page
    st.set_page_config(page_title="Assessment Generator", page_icon="📝")
    
    # Initialize session state variables
    if 'quiz_manager' not in st.session_state:
        st.session_state.quiz_manager = QuizManager()
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if "model" not in st.session_state:
        st.session_state.model = 'Gemma 2 Instruct'
    if "topic" not in st.session_state:
        st.session_state.topic = 'Psychometry'
    if "display" in st.session_state:
        st.session_state.display = False
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = [None]

    # Set page title
    st.title("Assessment Generator")

    # Create sidebar for quiz settings
    st.sidebar.header("Quiz Settings")

    dict = {
        # "QwQ 32B": "qwen-qwq-32b",
        # "DeepSeek R1 Distill Llama 70B": 'deepseek-r1-distill-llama-70b',
        "Gemma 2 Instruct": "gemma2-9b-it",
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "Llama 3.3 70B": "llama-3.3-70b-versatile",
        "Llama 3 70B": "llama3-70b-8192",
        "Llama 3 8B": "llama3-8b-8192",
        "Llama 4 Maverick 17B 128E": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "Llama 4 Scout 17B 16E": "meta-llama/llama-4-scout-17b-16e-instruct",
        # "Mistral Saba 24B": "mistral-saba-24b"
    }
    
    # API selection dropdown
    api_choice = st.sidebar.selectbox(
        "Select Model", 
        dict.keys(),
        index=0
    )


    # Question type selection
    # question_type = st.sidebar.selectbox(
    #     "Select Question Type", 
    #     ["Multiple Choice", "Fill in the Blank"], 
    #     index=0
    # )

    # Topic input field
    topic = st.sidebar.selectbox(
        "Select Topic", 
        ["Psychometry"], 
        index=0
    )


    # Number of questions input
    num_questions = st.sidebar.number_input(
        "Number of Questions", 
        min_value=0, 
        max_value=100, 
        value=10
    )

    if dict[api_choice] != st.session_state.model or topic != st.session_state.topic:
        st.session_state.quiz_manager = QuizManager()
        st.session_state.quiz_generated = False
        st.session_state.quiz_submitted = False
        st.session_state.model = dict[api_choice]
        st.session_state.topic = topic
        st.session_state.user_answers = [None] * num_questions

    if st.sidebar.button("Generate Questions"):
        if st.session_state.topic == "Psychometry":
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = [None] * num_questions
            generator = QuestionGenerator(dict[api_choice])
            st.session_state.quiz_generated = st.session_state.quiz_manager.generate_questions(
                generator, "Multiple Choice", num_questions
            )
            st.rerun()

    # 🚨 Move the rest OUTSIDE the button click
    if st.session_state.quiz_generated and st.session_state.quiz_manager.questions:
        st.header(st.session_state.topic + " Test")
        st.session_state.quiz_manager.attempt_quiz()

        # Submit quiz button
        if st.button("Submit Quiz"):
            st.session_state.quiz_manager.evaluate_quiz()
            st.session_state.quiz_submitted = True
            st.rerun()

    # Display results if quiz is submitted
    # Display results if quiz is submitted


    if st.session_state.quiz_submitted:
        st.header("📊 Psychometric Quiz Results")
        results_df = st.session_state.quiz_manager.generate_result_dataframe()
        
        if not results_df.empty:
            total_questions = len(results_df)
            avg_score = results_df['Score'].mean() * 100

            st.info(f"**Psychometric Profile Intensity:** {avg_score:.1f}% — based on {total_questions} insights")

            # Create two columns for side-by-side charts
            col1, col2 = st.columns(2)

            # --- Pie Chart in col1 ---
            with col1:
                st.subheader("🔹Score Label Distribution")
                label_counts = results_df['Label'].value_counts().reset_index()
                label_counts.columns = ['Label', 'Count']
                fig2 = px.pie(label_counts, names='Label', values='Count', title='Low / Moderate / High Distribution')
                st.plotly_chart(fig2, use_container_width=True)

            # --- Radar Chart in col2 ---
            with col2:
                st.subheader("🔹Dimension-wise Profile")

                # Prepare radar data
                categories = results_df['Dimension'].unique().tolist()
                values = results_df.groupby('Dimension')['Score'].mean().reindex(categories).fillna(0).tolist()
                values += values[:1]
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
                ax.plot(angles, values, linewidth=2, linestyle='solid', label='Average Score')
                ax.fill(angles, values, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_yticklabels([])
                ax.set_title("Dimension-wise Normalized Score Radar", y=1.08)
                ax.grid(True)

                st.pyplot(fig)

            # === Show Table with Color-Coding ===
            st.subheader("🔹 Detailed Breakdown")
            # styled_df = results_df.style.background_gradient(subset=['Score'], cmap='YlGnBu')
            # st.dataframe(styled_df, use_container_width=True)

            # === Show Per Question Breakdown ===
            for _, result in results_df.iterrows():
                question_num = result['question_number']
                st.markdown(f"#### Question {question_num}")
                st.write(f"**Question:** {result['question']}")
                st.write(f"**Your Answer:** {result['user_answer']}")
                st.write(f"**Inferred Dimension:** {result['Dimension']}")
                st.write(f"**Score:** {result['Score']:.2f} ({result['Label']})")
                st.write(f"**Reasoning:** {result['Reasoning']}")
                st.markdown("---")

            # === Save Results ===
            if st.button("💾 Save Results"):
                saved_file = st.session_state.quiz_manager.save_to_csv()
                if saved_file:
                    with open(saved_file, 'rb') as f:
                        st.download_button(
                            label="📥 Download CSV",
                            data=f.read(),
                            file_name=os.path.basename(saved_file),
                            mime='text/csv'
                        )
        else:
            st.warning("⚠️ No results available. Please complete the quiz first.")


# Entry point of the application
if __name__ == "__main__":
    main()