# Day 3: Async Python & API Basics
import asyncio
import time

async def fetch_ai_response(model_name: str, delay: int) -> str:
    print(f"[{time.strftime('%X')}] Requesting {model_name}...")
    await asyncio.sleep(delay)
    print(f"[{time.strftime('%X')}] Response received from {model_name}!")
    return f"Response from {model_name}"

async def main():
    start_time = time.time()
    
    results = await asyncio.gather(
        fetch_ai_response("Gemini Pro", 2),
        fetch_ai_response("GPT-4o", 3),
        fetch_ai_response("Claude 3.5 Sonnet", 1)
    )
    
    elapsed = time.time() - start_time
    print(f"\nAll API calls completed in {elapsed:.2f} seconds!")
    print("Results:", results)

if __name__ == "__main__":
    asyncio.run(main())