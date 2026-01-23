## Pydantic AI CLI Chatbot

Minimal proof-of-concept CLI app that uses a Pydantic AI agent with Anthropic, includes basic tools (read/write files, run bash, run python), and prints tool outputs in the terminal.

### Run

1. Set your Anthropic key in `.env`:
   - `ANTHROPIC_API_KEY=...`
2. Install dependencies:
   - `uv sync`
3. Start the app:
   - `uv run python src/app.py`
