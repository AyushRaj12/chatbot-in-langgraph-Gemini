# LangGraph Chatbot

A public Streamlit chatbot built with LangGraph and Google Gemini.

## Run locally

Use Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Replace the placeholder in `.env` with your Gemini API key, then run:

```bash
streamlit run streamlit_frontend.py
```

The local server should be available at `http://127.0.0.1:8501`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push this project to it. Keep `.env` and `.streamlit/secrets.toml` out of Git.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
3. Select the GitHub repository and branch.
4. Set the entrypoint to `streamlit_frontend.py`.
5. In **Advanced settings**, choose Python 3.12 and add these secrets:

   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key"
   GEMINI_MODEL = "gemini-3.8-flash"
   ```

6. Select **Deploy** and test the generated public URL.

Streamlit documents the current workflow in [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [Secrets management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management).

## Public deployment limits

Anyone with the public URL can consume your Gemini free-tier quota. The app limits message length, response length, and requests per browser session, but visitors can start new sessions. Once the free quota is exhausted, Gemini may throttle or reject requests until the quota resets. Free-tier prompts and responses may be used to improve Google products, so visitors should not submit sensitive information. Never put the API key in source code or commit it to Git.

## Current persistence behavior

Conversation state is isolated with a random ID for each browser session. It is held in process memory and can disappear whenever Streamlit restarts the app.
