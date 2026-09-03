# Unified Async LLM Client

Cliente async para OpenAI, Anthropic y Google (Gemini), con la misma interfaz, streaming y validación con Pydantic.

## Setup

Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

En Linux/macOS: `source .venv/bin/activate` y `cp .env.example .env`.

Completá las keys en `.env`. Ese archivo no se sube al repo.

## Variables de entorno

- `LLM_PROVIDER`: `openai`, `anthropic` o `google`
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`: la del proveedor que elijas
- Modelos opcionales: `OPENAI_MODEL` (`gpt-4o-mini`), `ANTHROPIC_MODEL` (`claude-3-5-sonnet-latest`), `GOOGLE_MODEL` (`gemini-3.6-flash`)

## Cómo probarlo

```powershell
python main.py
```

Hace la pregunta "¿Qué es la entropía?" en modo normal y en streaming. Si falla la API (key, red, rate limit), muestra un error y no corta el proceso.

Se agrego GOOGLE como proveedor para poder hacer una prueba con un proveedor con el cual tengo tokens.

Para cambiar de proveedor, editá `LLM_PROVIDER` en `.env` y volvé a correr el script.
