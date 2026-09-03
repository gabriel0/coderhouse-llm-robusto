# Unified Async LLM Client

Cliente asíncrono unificado para llamar a **OpenAI** o **Anthropic** con la misma interfaz. Soporta generación completa, streaming de tokens, validación con Pydantic y errores controlados (red, rate limit, API key inválida).

## Requisitos

- Python 3.12
- Una API key de OpenAI y/o Anthropic

## Instalación

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Linux / macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y completa las keys. El archivo `.env` no se versiona.

## Variables de entorno

| Variable | Obligatoria | Descripción |
| --- | --- | --- |
| `LLM_PROVIDER` | Sí | `openai` o `anthropic` |
| `OPENAI_API_KEY` | Si el proveedor es OpenAI | Key del dashboard de OpenAI |
| `ANTHROPIC_API_KEY` | Si el proveedor es Anthropic | Key del dashboard de Anthropic |
| `OPENAI_MODEL` | No | Default: `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | No | Default: `claude-3-5-sonnet-latest` |

## Cómo ejecutar la prueba

Desde la raíz del repositorio, con el venv activo y el `.env` configurado:

```bash
python main.py
```

El script pregunta **"¿Qué es la entropía?"** dos veces:

1. **Modo normal**: espera la respuesta completa y la imprime.
2. **Modo streaming**: imprime los tokens a medida que llegan (`yield` + `async for`).

Si falta la API key, hay un límite de cuota o un error de red, el proceso **no crashea**: se muestra un mensaje de error controlado.

Para probar el otro proveedor, cambia `LLM_PROVIDER` en `.env` y vuelve a correr `python main.py`.

## Estructura

| Archivo | Rol |
| --- | --- |
| `schemas.py` | Modelos Pydantic: `ChatMessage`, `ModelParams` (temperatura 0–2, `max_tokens`), `ModelResponse`, `LLMConfig` |
| `base.py` | `BaseLLMClient` abstracto (`generate` / `generate_stream`) + retry y mapeo de errores |
| `openai_client.py` | `OpenAIClient` con `AsyncOpenAI` |
| `anthropic_client.py` | `AnthropicClient` con `AsyncAnthropic` |
| `manager.py` | `AsyncLLMManager`: elige el proveedor según `LLM_PROVIDER` |
| `main.py` | Script de validación (modo normal + streaming) |
| `.env.example` | Plantilla de variables de entorno |

## Uso rápido en código

```python
import asyncio
from manager import AsyncLLMManager
from schemas import ChatMessage, ModelParams

async def demo() -> None:
    manager = AsyncLLMManager.from_env()
    messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]
    params = ModelParams(temperature=0.5, max_tokens=300)

    response = await manager.generate(messages, params)
    print(response.content)

    async for token in manager.generate_stream(messages, params):
        print(token, end="", flush=True)

asyncio.run(demo())
```
