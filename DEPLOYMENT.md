# Deploying Future AI Engineer to Streamlit Cloud

This guide will walk you through deploying your **AION Coach** app to the Streamlit Community Cloud.

## Prerequisites

1.  **GitHub Account**: You need a GitHub account to host your code.
2.  **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.

## Step 1: Push Code to GitHub

If you haven't already, you need to push your code to a new GitHub repository.

1.  **Create a New Repository** on GitHub (e.g., `future-ai-engineer`).
2.  **Initialize Git** (if not done) and push:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/future-ai-engineer.git
git push -u origin main
```

> [!NOTE]
> Make sure your `requirements.txt` file is in the root directory.

## Step 2: Deploy on Streamlit Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  Select your repository (`future-ai-engineer`), branch (`main`), and main file path (`app.py`).
4.  Click **"Deploy!"**.

## Step 3: Configure Secrets (API Key)

Your app needs the `GEMINI_API_KEY` to function. Since we shouldn't commit `.env` files to GitHub, we use Streamlit Secrets.

1.  Once deployed (or while it's building), go to your **App Dashboard** on Streamlit Cloud.
2.  Click the **"Settings"** (three dots) on your app card or the top right menu of the viewer.
3.  Select **"Settings"** -> **"Secrets"**.
4.  Paste your API key in TOML format:

```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

5.  Click **"Save"**.

The app should automatically detect the change and reload with the key connected!
