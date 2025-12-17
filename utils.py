import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Global client holder, but ideally we instantiate per request or use a singleton properly.
# For Streamlit, we can just instantiate simple client or store in session state, but here we keep it simple.
_client = None

def init_gemini(api_key=None):
    """
    Initialize the Gemini API client.
    """
    global _client
    key = api_key or os.getenv("GEMINI_API_KEY")
    if key:
        _client = genai.Client(api_key=key)
        return True
    return False

def get_gemini_response(prompt, system_instruction=None, model_name="gemini-2.5-flash"):
    """
    Get a response from Gemini using the new SDK.
    """
    global _client
    # Auto-init if not already done and env var exists
    if not _client:
        if os.getenv("GEMINI_API_KEY"):
             init_gemini()
        else:
             return "Error: Gemini API Key not set."

    try:
        config = {}
        if system_instruction:
            config['system_instruction'] = system_instruction
            
        if "JSON" in (system_instruction or "").upper() or "JSON" in prompt.upper():
             config['response_mime_type'] = "application/json"

        response = _client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"

def evaluate_submission(submission_text, context=""):
    """
    Evaluate a learner's submission using the AION Coach rubric.
    """
    rubric = """
    Evaluate the following submission based on:
    1. Correctness (0-4)
    2. Code Quality (0-4)
    3. Clarity/Docs (0-4)
    4. Efficiency (0-4)
    5. Testing (0-4)

    Provide:
    - 3 Strengths
    - 3 Critical Fixes
    - 1 Stretch Improvement
    - A remedial micro-task if any score is < 3.
    """
    prompt = f"{rubric}\n\nContext: {context}\n\nSubmission:\n{submission_text}"
    return get_gemini_response(prompt, system_instruction="You are AION Coach, an expert AI Engineering Trainer.")

def generate_quiz(topic, difficulty="intermediate"):
    """
    Generate a quiz for a given topic.
    """
    prompt = f"Generate a 5-question quiz (mixed MCQ and short answer) for the topic: {topic}. Difficulty: {difficulty}. Return in JSON format."
    return get_gemini_response(prompt, system_instruction="You are a strict technical interviewer.")
