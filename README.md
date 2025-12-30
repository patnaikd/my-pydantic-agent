## Pydantic AI Streamlit Chatbot

Minimal proof-of-concept Streamlit app that uses a Pydantic AI agent with Anthropic, includes basic tools (read/write files, run bash, run python), and shows tool outputs in the chat UI.

### Run

1. Set your Anthropic key in `.env`:
   - `ANTHROPIC_API_KEY=...`
2. Install dependencies:
   - `uv sync`
3. Start the app:
   - `uv run streamlit run src/app.py`
