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

REVERSE_SYSTEM_PROMPT = """You are a journalist converting satirical Onion-style headlines into straight, factual news headlines.
Rewrite the given sarcastic/satirical headline as a neutral news headline a real newspaper would publish.

Rules:
- Keep the same general topic and named entities where possible
- Use objective, journalistic tone — no irony, sarcasm, or humor
- Similar length to input; one headline only
- Output ONLY the transformed headline — no explanation, no prefix

Examples:

Input: Area Frog Deeply Upset Its Entire Existence Spent Hiding From Scientists
Output: Scientists discover new species of frog in Amazon rainforest

Input: Nation's Wealthy Reassured Everything Fine As Number Continues Going Up
Output: Stock market reaches record high amid economic uncertainty

Input: Area Man Vows To Remain Completely Unchanged By Sudden Acquisition Of Everything He Ever Wanted
Output: Local man wins $1 million lottery jackpot

Input: Researchers Confirm Teenagers Doing Thing Every Adult Has Told Them Not To Do
Output: New study links excessive screen time to poor sleep in teenagers

Input: Nation's Potholes Given Formal 10-Year Countdown To Possible Extinction
Output: Government approves $1.2 trillion infrastructure spending bill

Input: Company Celebrates Best Year Ever By Removing Humans Who Made It Possible
Output: Tech company lays off 10,000 workers after record profits

Input: Scientists Release Annual Report Confirming Everything Still Going Exactly As Badly As Predicted
Output: Climate scientists warn of accelerating ice sheet melt

Input: Two People Who Spent $4 Million On Wedding Reportedly Growing Apart
Output: Celebrity couple announces divorce after 2 years of marriage

Input: Doctors Baffled By Product That Finally Solves The One Problem Doctors Have Always Said Requires Effort
Output: New diet pill promises dramatic weight loss with no exercise

Input: City Leaders Celebrate Bold Vision For Making Downtown Slightly Less Miserable To Drive Through
Output: City council approves new downtown parking garage

Input: Nation's Travelers Reminded That Getting From One Place To Another Remains A Privilege Not A Right
Output: Airline cancels hundreds of flights due to staffing shortage

Input: Researchers Confirm People Perform Better Without Commuting Two Hours To Sit In Different Chair
Output: Study finds remote workers are more productive than office workers

Input: Company That Built Entire Business On Selling Your Data Introduces Button That Does Nothing
Output: Social media company introduces new privacy settings

Input: Drug Maker Announces Compassionate New Program Allowing Diabetics To Simply Afford Less
Output: Pharmaceutical company raises insulin price by 400 percent

Input: Local Man Running For Office Identifies Problem That Has Stumped Everyone Else For 50 Years
Output: Politician promises to fix healthcare if elected

Input: Researchers Announce Breakthrough That Will Definitely Be Available To Consumers Within 15 Years
Output: Scientists develop new battery that charges in 5 minutes

Input: Americans Flood Last Remaining Natural Spaces To Escape Effects Of Flooding Last Remaining Natural Spaces
Output: National park reports record visitor numbers this summer

Input: Restaurant That Pioneered Mechanically Separated Meat Now Offering Consumers Chance To Feel Better About Themselves
Output: Fast food chain introduces plant-based burger option

Input: Real Estate Experts Confirm Dream Of Homeownership Still Technically Legal To Have
Output: Housing prices reach all-time high in major cities

Input: Scientists Discover That Thing Millions Were Already Doing Every Morning Is Fine Actually
Output: Study shows coffee consumption linked to longer lifespan

Input: Mayor Announces Bold New Initiative To Reduce Visibility Of Problem By Half
Output: City mayor unveils plan to reduce homelessness by 50 percent

Input: Nation's Gas Stations Prepare Heartfelt Farewell Tour Nobody Asked For
Output: Electric vehicle sales surpass gasoline cars for first time

Input: Institution Dedicated To Expanding Minds Expands Price Tag Once Again
Output: University raises tuition fees by 8 percent citing rising costs

Input: Area Chef Relieved To Learn Decades Of Hard Work Were Not Entirely Wasted
Output: Local restaurant receives Michelin star after 30 years in business

Input: Researchers Assure Public That Plastic Now Coursing Through Their Veins Is Probably Fine
Output: Scientists warn that microplastics found in human blood

Input: Economists Agree Best Way To Help Struggling Americans Is To Make Borrowing Money More Expensive
Output: Federal Reserve raises interest rates to combat inflation

Input: Tech Startup Solves Human Connection Problem By Removing Humans From Equation Entirely
Output: New app promises to cure loneliness through AI companionship

Input: Officials Remind Citizens That Being Watched At All Times Is Actually Very Comforting
Output: City installs surveillance cameras on every street corner for public safety

Input: Company That Replaced Main Street With Giant Parking Lot Now Being Replaced By Website
Output: Major retailer announces it will close 200 stores nationwide

Input: Researchers Publish Findings That Will Give Many People Something New To Worry About
Output: Scientists confirm stress is bad for your health

Input: Climate Officials Celebrate Impressive New Number Achieved For Seventh Consecutive Year
Output: Nation marks another year of record-breaking summer temperatures

Input: Town Of 4,000 Receives Tank To Handle Whatever May Come
Output: Local police department receives military-grade equipment grant

Input: Researchers Confirm Going Outside Still Technically An Option Available To Everyone
Output: Study reveals people are happier when spending time outdoors

Input: Company That Promised To Replace Cable Announces It Has Become Cable
Output: Streaming service raises monthly subscription price by $3"""


def load_dataset(path: str, is_sarcastic_filter: int = 0) -> list[str]:
    headlines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("is_sarcastic") == is_sarcastic_filter:
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


async def call_api(client: AsyncOpenAI, headline: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
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
    direction: str = "to-sarcastic",
    system_prompt: str = SYSTEM_PROMPT,
) -> None:
    async with semaphore:
        try:
            result = await call_api(client, headline, system_prompt)
            if direction == "to-neutral":
                record = {"input": result, "output": headline}
            else:
                record = {"input": headline, "output": result}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            counters["ok"] += 1
        except Exception as e:
            print(f"WARNING: failed for {headline!r}: {e}", file=sys.stderr)
            counters["failed"] += 1
        pbar.update(1)


async def run_deepseek(
    todo: list[str], output_path: str, concurrency: int, api_key: str, batch_size: int,
    direction: str = "to-sarcastic", system_prompt: str = SYSTEM_PROMPT,
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
                    *[
                        process_headline(h, semaphore, client, f, pbar, counters, direction, system_prompt)
                        for h in chunk
                    ],
                    return_exceptions=True,
                )

    print(f"\nDone: {counters['ok']} succeeded, {counters['failed']} failed.")


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

async def create_gemini_cache(client, system_prompt: str = SYSTEM_PROMPT) -> "str | None":
    from google.genai import types
    try:
        cache = await client.aio.caches.create(
            model=GEMINI_MODEL,
            config=types.CreateCachedContentConfig(
                system_instruction=system_prompt,
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
    todo: list[str], output_path: str, api_key: str, concurrency: int,
    direction: str = "to-sarcastic", system_prompt: str = SYSTEM_PROMPT,
) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cache_name = await create_gemini_cache(client, system_prompt)

    semaphore = asyncio.Semaphore(concurrency)
    counters = {"ok": 0, "failed": 0}

    async def process_one(headline: str, f) -> None:
        async with semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    if cache_name:
                        cfg = types.GenerateContentConfig(cached_content=cache_name)
                    else:
                        cfg = types.GenerateContentConfig(system_instruction=system_prompt)
                    response = await client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=headline,
                        config=cfg,
                    )
                    text = response.text.strip()
                    if direction == "to-neutral":
                        record = {"input": text, "output": headline}
                    else:
                        record = {"input": headline, "output": text}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    counters["ok"] += 1
                    return
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(BASE_DELAY * 2 ** attempt)
                    else:
                        print(f"WARNING: failed for {headline!r}: {e}", file=sys.stderr)
                        counters["failed"] += 1

    try:
        with open(output_path, "a") as f:
            with tqdm(total=len(todo), desc="Generating") as pbar:
                async def task(h):
                    await process_one(h, f)
                    pbar.update(1)
                await asyncio.gather(*[task(h) for h in todo])
    finally:
        if cache_name:
            try:
                await client.aio.caches.delete(name=cache_name)
                print(f"Context cache deleted: {cache_name}", file=sys.stderr)
            except Exception as e:
                print(f"WARNING: could not delete cache {cache_name}: {e}", file=sys.stderr)

    print(f"\nDone: {counters['ok']} succeeded, {counters['failed']} failed.")


async def run_gemini_submit(
    todo: list[str], output_path: str, api_key: str, batch_size: int,
    direction: str = "to-sarcastic", system_prompt: str = SYSTEM_PROMPT,
) -> None:
    from google import genai

    client = genai.Client(api_key=api_key)
    state_path = output_path + ".state.json"

    # Load existing state so we can append without losing prior jobs
    state: dict = {"chunks": [], "direction": direction}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        # Preserve direction from first submit if not yet set
        if "direction" not in state:
            state["direction"] = direction

    # Skip headlines already submitted in a prior (interrupted) run
    already_submitted = {h for ci in state["chunks"] for h in ci["headlines"]}
    remaining = [h for h in todo if h not in already_submitted]
    if not remaining:
        print("All chunks already submitted. Run with --mode collect to retrieve results.")
        return

    chunks = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]

    # Don't use context caching in submit mode: the cache would be deleted before
    # the batch jobs are processed asynchronously, causing CachedContent not found errors.
    for idx, chunk in enumerate(chunks):
        label = f"sarcasm-batch-{idx + 1}-of-{len(chunks)}"
        print(f"\n[Chunk {idx + 1}/{len(chunks)}] Submitting {len(chunk)} headlines…", file=sys.stderr)

        for attempt in range(MAX_RETRIES):
            try:
                job = await submit_gemini_batch(client, chunk, None, label)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * 2 ** attempt
                    print(f"  Submission error ({e}), retrying in {delay:.0f}s…", file=sys.stderr)
                    await asyncio.sleep(delay)
                else:
                    raise

        state["chunks"].append({
            "job_name": job.name,
            "headlines": chunk,
            "status": "pending",
        })
        with open(state_path, "w") as f:
            json.dump(state, f)

    print(f"\nSubmitted {len(chunks)} batch job(s). State saved to: {state_path}")
    print("Run with --mode collect to retrieve results when jobs complete.")


async def run_gemini_collect(output_path: str, api_key: str) -> None:
    from google import genai

    state_path = output_path + ".state.json"
    if not os.path.exists(state_path):
        print(f"ERROR: State file not found: {state_path}", file=sys.stderr)
        print("Run with --mode submit first.", file=sys.stderr)
        sys.exit(1)

    with open(state_path) as f:
        state = json.load(f)

    client = genai.Client(api_key=api_key)
    already_done = load_already_done(output_path)
    direction = state.get("direction", "to-sarcastic")

    total_ok = 0
    total_failed = 0
    total_skipped = 0

    with open(output_path, "a") as out_f:
        for idx, chunk_info in enumerate(state["chunks"]):
            if chunk_info["status"] == "collected":
                print(f"[Chunk {idx + 1}] Already collected, skipping.", file=sys.stderr)
                continue

            job_name = chunk_info["job_name"]
            headlines = chunk_info["headlines"]
            print(f"\n[Chunk {idx + 1}] Polling job {job_name}…", file=sys.stderr)

            job = await client.aio.batches.get(name=job_name)
            job = await poll_gemini_batch(client, job)

            state_name = job.state.name if hasattr(job.state, "name") else str(job.state)
            if state_name in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                print(f"  Job {job_name} ended with state {state_name}, marking as failed.", file=sys.stderr)
                chunk_info["status"] = "failed"
            else:
                successes, failures = parse_gemini_batch_results(job, headlines)

                written = 0
                for headline, result in successes:
                    if headline in already_done:
                        total_skipped += 1
                        continue
                    if direction == "to-neutral":
                        record = {"input": result, "output": headline}
                    else:
                        record = {"input": headline, "output": result}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    already_done.add(headline)
                    written += 1
                out_f.flush()

                total_ok += written
                total_failed += len(failures)
                chunk_info["status"] = "collected"
                print(
                    f"  Chunk {idx + 1}: {written} written, {len(failures)} failed, {total_skipped} skipped (already done).",
                    file=sys.stderr,
                )

            with open(state_path, "w") as f:
                json.dump(state, f)

    pending = sum(1 for c in state["chunks"] if c["status"] == "pending")
    print(f"\nDone: {total_ok} written, {total_failed} failed, {total_skipped} skipped.")
    if pending:
        print(f"{pending} chunk(s) still pending — re-run --mode collect to retry.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sarcastic headlines via DeepSeek-V3 or Gemini 2.5 Flash"
    )
    parser.add_argument("--dataset", default="Sarcasm_Headlines_Dataset.json", help="Input JSONL dataset")
    parser.add_argument("--output", default=None, help="Output JSONL file (default depends on --direction)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N remaining headlines")
    parser.add_argument("--concurrency", type=int, default=10, help="Max simultaneous API calls (DeepSeek only)")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "gemini"], help="API provider")
    parser.add_argument("--batch-size", type=int, default=500, help="Chunk size for both providers")
    parser.add_argument(
        "--mode",
        default="run",
        choices=["run", "submit", "collect"],
        help="run: submit+poll inline (default); submit: submit batch jobs and exit immediately; collect: retrieve results from previously submitted jobs (Gemini only)",
    )
    parser.add_argument(
        "--direction",
        default="to-sarcastic",
        choices=["to-sarcastic", "to-neutral"],
        help="to-sarcastic: generate sarcastic from neutral headlines (default); to-neutral: generate neutral from sarcastic headlines",
    )
    args = parser.parse_args()

    # Resolve default output path based on direction
    if args.output is None:
        args.output = "synthetic_neutral.jsonl" if args.direction == "to-neutral" else "synthetic_sarcastic.jsonl"

    system_prompt = REVERSE_SYSTEM_PROMPT if args.direction == "to-neutral" else SYSTEM_PROMPT
    is_sarcastic_filter = 1 if args.direction == "to-neutral" else 0

    if args.mode == "collect":
        if args.provider != "gemini":
            print("ERROR: --mode collect is only supported for --provider gemini", file=sys.stderr)
            sys.exit(1)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_gemini_collect(args.output, api_key))
        return

    all_headlines = load_dataset(args.dataset, is_sarcastic_filter)
    done_set = load_already_done(args.output)

    todo = [h for h in all_headlines if h not in done_set]
    if args.limit is not None:
        todo = todo[: args.limit]

    label = "sarcastic" if is_sarcastic_filter else "non-sarcastic"
    print(f"Dataset: {len(all_headlines)} {label} headlines")
    print(f"Already done: {len(done_set)}")
    print(f"To process: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    if args.provider == "deepseek":
        if args.mode != "run":
            print("ERROR: --mode submit is only supported for --provider gemini", file=sys.stderr)
            sys.exit(1)
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("ERROR: DEEPSEEK_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_deepseek(todo, args.output, args.concurrency, api_key, args.batch_size, args.direction, system_prompt))
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
            sys.exit(1)
        if args.mode == "submit":
            asyncio.run(run_gemini_submit(todo, args.output, api_key, args.batch_size, args.direction, system_prompt))
        else:
            asyncio.run(run_gemini(todo, args.output, api_key, args.concurrency, args.direction, system_prompt))


if __name__ == "__main__":
    main()
