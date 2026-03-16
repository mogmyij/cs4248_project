#!/usr/bin/env python3
"""Generate synthetic sarcastic headlines using DeepSeek-V3 or Gemini 2.5 Flash."""

import argparse
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL   = "https://api.deepseek.com"
MODEL_NAME     = "deepseek-chat"
MAX_RETRIES    = 3
BASE_DELAY     = 1.0
TEMPERATURE    = 0.9
MAX_TOKENS     = 100

GEMINI_MODEL   = "gemini-2.5-flash"
CACHE_TTL      = "7200s"
POLL_INTERVAL  = 15  # seconds between batch status polls

SYSTEM_PROMPT = """You are a headline writer for a satirical news publication in the style of The Onion.
Transform the given genuine news headline into a sarcastic, satirical version.

Rules:
- Keep the same general topic and named entities where possible
- Use deadpan, mock-serious tone — absurdity in content, not exclamation marks
- Do NOT use words like "sarcastic", "ironically", or "satirically"
- Similar length to input; one headline only
- Output ONLY the transformed headline — no explanation, no prefix

Examples:

Input: Scientists discover new species of frog in Amazon rainforest
Output: Area Frog Deeply Upset Its Entire Existence Spent Hiding From Scientists

Input: Stock market reaches record high amid economic uncertainty
Output: Nation's Wealthy Reassured Everything Fine As Number Continues Going Up

Input: Local man wins $1 million lottery jackpot
Output: Area Man Vows To Remain Completely Unchanged By Sudden Acquisition Of Everything He Ever Wanted

Input: New study links excessive screen time to poor sleep in teenagers
Output: Researchers Confirm Teenagers Doing Thing Every Adult Has Told Them Not To Do

Input: Government approves $1.2 trillion infrastructure spending bill
Output: Nation's Potholes Given Formal 10-Year Countdown To Possible Extinction

Input: Tech company lays off 10,000 workers after record profits
Output: Company Celebrates Best Year Ever By Removing Humans Who Made It Possible

Input: Climate scientists warn of accelerating ice sheet melt
Output: Scientists Release Annual Report Confirming Everything Still Going Exactly As Badly As Predicted

Input: Celebrity couple announces divorce after 2 years of marriage
Output: Two People Who Spent $4 Million On Wedding Reportedly Growing Apart

Input: New diet pill promises dramatic weight loss with no exercise
Output: Doctors Baffled By Product That Finally Solves The One Problem Doctors Have Always Said Requires Effort

Input: City council approves new downtown parking garage
Output: City Leaders Celebrate Bold Vision For Making Downtown Slightly Less Miserable To Drive Through

Input: Airline cancels hundreds of flights due to staffing shortage
Output: Nation's Travelers Reminded That Getting From One Place To Another Remains A Privilege Not A Right

Input: Study finds remote workers are more productive than office workers
Output: Researchers Confirm People Perform Better Without Commuting Two Hours To Sit In Different Chair

Input: Social media company introduces new privacy settings
Output: Company That Built Entire Business On Selling Your Data Introduces Button That Does Nothing

Input: Pharmaceutical company raises insulin price by 400 percent
Output: Drug Maker Announces Compassionate New Program Allowing Diabetics To Simply Afford Less

Input: Politician promises to fix healthcare if elected
Output: Local Man Running For Office Identifies Problem That Has Stumped Everyone Else For 50 Years

Input: Scientists develop new battery that charges in 5 minutes
Output: Researchers Announce Breakthrough That Will Definitely Be Available To Consumers Within 15 Years

Input: National park reports record visitor numbers this summer
Output: Americans Flood Last Remaining Natural Spaces To Escape Effects Of Flooding Last Remaining Natural Spaces

Input: Fast food chain introduces plant-based burger option
Output: Restaurant That Pioneered Mechanically Separated Meat Now Offering Consumers Chance To Feel Better About Themselves

Input: Housing prices reach all-time high in major cities
Output: Real Estate Experts Confirm Dream Of Homeownership Still Technically Legal To Have

Input: Study shows coffee consumption linked to longer lifespan
Output: Scientists Discover That Thing Millions Were Already Doing Every Morning Is Fine Actually

Input: City mayor unveils plan to reduce homelessness by 50 percent
Output: Mayor Announces Bold New Initiative To Reduce Visibility Of Problem By Half

Input: Electric vehicle sales surpass gasoline cars for first time
Output: Nation's Gas Stations Prepare Heartfelt Farewell Tour Nobody Asked For

Input: University raises tuition fees by 8 percent citing rising costs
Output: Institution Dedicated To Expanding Minds Expands Price Tag Once Again

Input: Local restaurant receives Michelin star after 30 years in business
Output: Area Chef Relieved To Learn Decades Of Hard Work Were Not Entirely Wasted

Input: Scientists warn that microplastics found in human blood
Output: Researchers Assure Public That Plastic Now Coursing Through Their Veins Is Probably Fine

Input: Federal Reserve raises interest rates to combat inflation
Output: Economists Agree Best Way To Help Struggling Americans Is To Make Borrowing Money More Expensive

Input: New app promises to cure loneliness through AI companionship
Output: Tech Startup Solves Human Connection Problem By Removing Humans From Equation Entirely

Input: City installs surveillance cameras on every street corner for public safety
Output: Officials Remind Citizens That Being Watched At All Times Is Actually Very Comforting

Input: Major retailer announces it will close 200 stores nationwide
Output: Company That Replaced Main Street With Giant Parking Lot Now Being Replaced By Website

Input: Scientists confirm stress is bad for your health
Output: Researchers Publish Findings That Will Give Many People Something New To Worry About

Input: Nation marks another year of record-breaking summer temperatures
Output: Climate Officials Celebrate Impressive New Number Achieved For Seventh Consecutive Year

Input: Local police department receives military-grade equipment grant
Output: Town Of 4,000 Receives Tank To Handle Whatever May Come

Input: Study reveals people are happier when spending time outdoors
Output: Researchers Confirm Going Outside Still Technically An Option Available To Everyone

Input: Streaming service raises monthly subscription price by $3
Output: Company That Promised To Replace Cable Announces It Has Become Cable"""


def load_dataset(path: str) -> list[str]:
    headlines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("is_sarcastic") == 0:
                    headlines.append(item["headline"])
            except (json.JSONDecodeError, KeyError):
                pass
    return headlines


def load_already_done(path: str) -> set[str]:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                done.add(item["input"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


async def call_api(client: AsyncOpenAI, headline: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": headline},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_DELAY * 2 ** attempt)
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BASE_DELAY * 2 ** attempt)
            else:
                raise
        except APIConnectionError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_DELAY * 2 ** attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries for headline: {headline!r}")


async def process_headline(
    headline: str,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    f,
    pbar: tqdm,
    counters: dict,
) -> None:
    async with semaphore:
        try:
            sarcastic = await call_api(client, headline)
            f.write(json.dumps({"input": headline, "output": sarcastic}, ensure_ascii=False) + "\n")
            f.flush()
            counters["ok"] += 1
        except Exception as e:
            print(f"WARNING: failed for {headline!r}: {e}", file=sys.stderr)
            counters["failed"] += 1
        pbar.update(1)


async def run_deepseek(
    todo: list[str], output_path: str, concurrency: int, api_key: str, batch_size: int
) -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    counters = {"ok": 0, "failed": 0}

    chunks = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]

    with open(output_path, "a") as f:
        for idx, chunk in enumerate(chunks):
            desc = f"Chunk {idx + 1}/{len(chunks)}"
            with tqdm(total=len(chunk), desc=desc) as pbar:
                await asyncio.gather(
                    *[process_headline(h, semaphore, client, f, pbar, counters) for h in chunk],
                    return_exceptions=True,
                )

    print(f"\nDone: {counters['ok']} succeeded, {counters['failed']} failed.")


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

async def create_gemini_cache(client) -> "str | None":
    from google.genai import types
    try:
        cache = await client.aio.caches.create(
            model=GEMINI_MODEL,
            config=types.CreateCachedContentConfig(
                system_instruction=SYSTEM_PROMPT,
                ttl=CACHE_TTL,
                display_name="sarcasm-sysprompt",
            ),
        )
        print(f"Context cache created: {cache.name}", file=sys.stderr)
        return cache.name
    except Exception as e:
        print(
            f"WARNING: cache creation failed ({e}); using inline system prompt",
            file=sys.stderr,
        )
        return None


async def submit_gemini_batch(client, headlines: list[str], cache_name: "str | None", label: str):
    from google.genai import types

    requests = []
    for headline in headlines:
        if cache_name:
            cfg = types.GenerateContentConfig(cached_content=cache_name)
        else:
            cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)

        requests.append(
            types.InlinedRequest(
                contents=headline,
                metadata={"headline": headline},
                config=cfg,
            )
        )

    job = await client.aio.batches.create(
        model=GEMINI_MODEL,
        src=requests,
        config=types.CreateBatchJobConfig(display_name=label),
    )
    print(f"Batch job submitted: {job.name}", file=sys.stderr)
    return job


async def poll_gemini_batch(client, job):
    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
    }
    while True:
        job = await client.aio.batches.get(name=job.name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        print(f"\r  Batch {job.name} — state: {state}   ", end="", flush=True, file=sys.stderr)
        if state in terminal_states:
            print(file=sys.stderr)
            return job
        await asyncio.sleep(POLL_INTERVAL)


def parse_gemini_batch_results(job, chunk: list[str]) -> "tuple[list, list]":
    successes = []
    failures = []

    responses = getattr(job.dest, "inlined_responses", None) or []
    for ir in responses:
        headline = (ir.metadata or {}).get("headline", "")
        if ir.error:
            print(f"WARNING: batch error for {headline!r}: {ir.error}", file=sys.stderr)
            failures.append(headline)
            continue
        text = ""
        try:
            text = ir.response.text.strip() if ir.response else ""
        except Exception:
            pass
        if not text:
            finish = ""
            try:
                finish = ir.response.candidates[0].finish_reason if ir.response else ""
            except Exception:
                pass
            print(
                f"WARNING: empty response for {headline!r} (finish_reason={finish})",
                file=sys.stderr,
            )
            failures.append(headline)
            continue
        successes.append((headline, text))

    return successes, failures


async def run_gemini(
    todo: list[str], output_path: str, api_key: str, batch_size: int
) -> None:
    from google import genai

    client = genai.Client(api_key=api_key)

    cache_name = await create_gemini_cache(client)

    total_ok = 0
    total_failed = 0
    chunks = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]

    try:
        with open(output_path, "a") as f:
            for idx, chunk in enumerate(chunks):
                label = f"sarcasm-batch-{idx + 1}-of-{len(chunks)}"
                print(f"\n[Chunk {idx + 1}/{len(chunks)}] Submitting {len(chunk)} headlines…", file=sys.stderr)

                job = await submit_gemini_batch(client, chunk, cache_name, label)
                job = await poll_gemini_batch(client, job)

                successes, failures = parse_gemini_batch_results(job, chunk)

                for headline, sarcastic in successes:
                    f.write(
                        json.dumps({"input": headline, "output": sarcastic}, ensure_ascii=False) + "\n"
                    )
                f.flush()

                total_ok += len(successes)
                total_failed += len(failures)
                print(
                    f"  Chunk {idx + 1}: {len(successes)} ok, {len(failures)} failed.",
                    file=sys.stderr,
                )
    finally:
        if cache_name:
            try:
                await client.aio.caches.delete(name=cache_name)
                print(f"Context cache deleted: {cache_name}", file=sys.stderr)
            except Exception as e:
                print(f"WARNING: could not delete cache {cache_name}: {e}", file=sys.stderr)

    print(f"\nDone: {total_ok} succeeded, {total_failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sarcastic headlines via DeepSeek-V3 or Gemini 2.5 Flash"
    )
    parser.add_argument("--dataset", default="Sarcasm_Headlines_Dataset.json", help="Input JSONL dataset")
    parser.add_argument("--output", default="synthetic_sarcastic.jsonl", help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Process only N remaining headlines")
    parser.add_argument("--concurrency", type=int, default=10, help="Max simultaneous API calls (DeepSeek only)")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "gemini"], help="API provider")
    parser.add_argument("--batch-size", type=int, default=500, help="Chunk size for both providers")
    args = parser.parse_args()

    all_headlines = load_dataset(args.dataset)
    done_set = load_already_done(args.output)

    todo = [h for h in all_headlines if h not in done_set]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"Dataset: {len(all_headlines)} non-sarcastic headlines")
    print(f"Already done: {len(done_set)}")
    print(f"To process: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    if args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("ERROR: DEEPSEEK_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_deepseek(todo, args.output, args.concurrency, api_key, args.batch_size))
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_gemini(todo, args.output, api_key, args.batch_size))


if __name__ == "__main__":
    main()
