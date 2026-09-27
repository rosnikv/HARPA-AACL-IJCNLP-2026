import os, asyncio, json, random, time
from pathlib import Path
from datasets import load_from_disk
from anthropic import AsyncAnthropic

# ---------------- Setup ----------------
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("Missing environment variable: ANTHROPIC_API_KEY")

OUTPUT_DIR = Path("smoke_test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "batch_results.jsonl"
INPUT_META_FILE = OUTPUT_DIR / "corresponding_input_ds.jsonl"  # true JSONL

dataset_path = "rmr1_traceGen_dataset"
ds = load_from_disk(dataset_path)
random.seed(42)

client = AsyncAnthropic(api_key=API_KEY)
# MODEL = "claude-3-7-sonnet-20250219"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192
TEMPERATURE = 0.0
CONCURRENCY = 2  # tweak if you want

# ------------- Build requests + save input metadata -------------
requests = []
with open(INPUT_META_FILE, "w", encoding="utf-8") as fmeta:
    for idx, item in enumerate(ds):
        system_prompt = None
        messages = []
        for m in item["context_messages"]:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                messages.append({"role": m["role"], "content": m["content"]})

        cid = f"task-{idx}"
        req = {"custom_id": cid, "messages": messages, "system": system_prompt}
        requests.append(req)

        # write input metadata as JSONL (one object per line)
        fmeta.write(json.dumps({
            "custom_id": cid,
            "winner": item.get("winner"),
            "context_messages": item.get("context_messages"),
        }) + "\n")

# ---------------- Helpers ----------------
def load_done_ids(path: Path):
    """Read existing JSONL and return custom_ids that already succeeded."""
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("status") == "succeeded":
                        done.add(obj.get("custom_id"))
                except Exception:
                    # ignore corrupt lines
                    pass
    return done

sem = asyncio.Semaphore(CONCURRENCY)

async def run_one(req, retries: int = 3, base_delay: float = 2.0):
    """Call Anthropic with retries; returns a result dict (succeeded/errored)."""
    cid, msgs, sys = req["custom_id"], req["messages"], req["system"]
    for attempt in range(retries):
        try:
            async with sem:
                kwargs = dict(model=MODEL, max_tokens=MAX_TOKENS, messages=msgs, temperature=TEMPERATURE)
                if sys:
                    kwargs["system"] = sys
                resp = await client.messages.create(**kwargs)

            text = resp.content[0].text if getattr(resp, "content", None) else ""
            print("{} → {}".format(cid, text[:200].replace('\n', ' ')))
            usage = getattr(resp, "usage", None)
            usage = usage.model_dump() if usage and hasattr(usage, "model_dump") else (usage or None)

            raw = resp.model_dump() if hasattr(resp, "model_dump") else getattr(resp, "__dict__", {})

            return {
                "custom_id": cid,
                "status": "succeeded",
                "response": {"text": text, "usage": usage, "raw": raw},
            }
        except Exception as e:
            if attempt < retries - 1:
                delay = (base_delay ** attempt) + random.random()
                print(f"[WARN] {cid} attempt {attempt+1}/{retries} failed: {e} — retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
            else:
                print(f"[ERROR] {cid} final failure: {e}")
                return {"custom_id": cid, "status": "errored", "error": repr(e)}

# ---------------- Run + save incrementally + resume ----------------
async def main():
    done_ids = load_done_ids(OUTPUT_FILE)
    pending = [r for r in requests if r["custom_id"] not in done_ids]

    total = len(requests)
    print(f"Total tasks: {total} | already done: {len(done_ids)} | pending: {len(pending)}")

    if not pending:
        print("✔ Nothing to do — all tasks already completed.")
        return

    # ensure file exists; we will append results as each finishes
    if not OUTPUT_FILE.exists():
        open(OUTPUT_FILE, "w", encoding="utf-8").close()

    started_at = time.time()
    tasks = [asyncio.create_task(run_one(r)) for r in pending]

    written = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for fut in asyncio.as_completed(tasks):
            res = await fut
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            os.fsync(fout.fileno())  # durability on crash/power loss
            written += 1
            if written % 10 == 0:
                elapsed = time.time() - started_at
                print(f"[PROGRESS] wrote {written}/{len(pending)} new results in {elapsed:.1f}s")

    print("✔ Async run complete. Results saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())
