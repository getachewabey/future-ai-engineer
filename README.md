# AION Coach - Future AI Engineer

This project is a 24-week (168-day) guided curriculum to become a job-ready AI Engineer.
It features a Streamlit-based "Trainer" app that provides daily instructions, tracks progress, and integrates with Google Gemini for AI-powered tutoring and code review.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file in this directory and add your Google API Key:
    ```
    GEMINI_API_KEY=your_key_here
    ```
    (Or enter it in the App Settings)

3.  **Run the Trainer App**:
    ```bash
    streamlit run app.py
    ```

## Structure
- `app.py`: Main application entry point.
- `curriculum.py`: Contains the full 168-day curriculum data.
- `utils.py`: Helper functions for Gemini integration and progress tracking.
