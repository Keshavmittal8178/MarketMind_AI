# ☁️ MarketMind AI — Cloud Deployment Guide

This document outlines the step-by-step instructions for deploying MarketMind AI to the cloud, satisfying **Requirement 11: Deploy your project on a cloud platform**.

---

## Option 1: Streamlit Community Cloud (Recommended — Free & 2 Minutes)

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Standardized MarketMind AI Multi-Agent System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/MarketMind_AI.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Sign in with your GitHub account.
   - Click **"New App"**.
   - Select your repository (`MarketMind_AI`), branch (`main`), and set the Main file path to:
     ```
     dashboard/app.py
     ```
   - Click **"Deploy"**!

3. **Configure Environment Secrets (Optional):**
   - In your app settings on Streamlit Cloud, go to **Secrets** and add:
     ```toml
     GROQ_API_KEY = "your_groq_api_key"
     ANTHROPIC_API_KEY = "your_anthropic_api_key"
     OPENAI_API_KEY = "your_openai_api_key"
     ```
   - If no keys are provided, the system seamlessly runs in labeled resilient offline mode.

---

## Option 2: Docker Container Deployment (AWS / GCP Cloud Run / Render)

1. **Build the Docker Image locally:**
   ```bash
   docker build -t marketmind-ai:latest .
   ```

2. **Run Container Locally to Test:**
   ```bash
   docker run -p 8501:8501 marketmind-ai:latest
   ```
   Open `http://localhost:8501` in your browser.

3. **Deploy to Google Cloud Run:**
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/marketmind-ai
   gcloud run deploy marketmind-ai --image gcr.io/<PROJECT_ID>/marketmind-ai --platform managed --port 8501 --allow-unauthenticated
   ```

4. **Deploy to Render:**
   - Connect your GitHub repo to [Render](https://render.com).
   - Select **Web Service** -> **Docker** runtime.
   - Port: `8501`.
   - Click **Deploy Web Service**.
