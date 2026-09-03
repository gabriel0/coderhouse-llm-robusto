import asyncio
import sys

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelParams


PROMPT = "¿Qué es la entropía?"


async def run_normal(manager: AsyncLLMManager, messages: list[ChatMessage], params: ModelParams) -> None:
    print("=== Modo normal ===")
    response = await manager.generate(messages, params)
    if response.success:
        print(response.content)
        if response.finish_reason:
            print(f"\n[finish_reason={response.finish_reason} | model={response.model}]")
    else:
        print(f"Error controlado: {response.error}")


async def run_streaming(manager: AsyncLLMManager, messages: list[ChatMessage], params: ModelParams) -> None:
    print("\n=== Modo streaming ===")
    async for token in manager.generate_stream(messages, params):
        print(token, end="", flush=True)
    print()


async def main() -> None:
    load_dotenv()
    try:
        manager = AsyncLLMManager.from_env()
    except ValueError as error:
        print(f"No se puede iniciar el cliente: {error}")
        sys.exit(1)

    print(f"Proveedor activo: {manager.provider.value}\n")
    messages = [ChatMessage(role="user", content=PROMPT)]
    params = ModelParams(temperature=0.5, max_tokens=300)

    await run_normal(manager, messages, params)
    await run_streaming(manager, messages, params)


if __name__ == "__main__":
    asyncio.run(main())
