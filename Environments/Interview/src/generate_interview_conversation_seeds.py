#!/usr/bin/env python
import argparse
import importlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


THIS_FILE = Path(__file__).resolve()
INTERVIEW_SRC = THIS_FILE.parent
INTERVIEW_ROOT = INTERVIEW_SRC.parent
REPO_ROOT = INTERVIEW_ROOT.parent

if str(INTERVIEW_SRC) not in sys.path:
    sys.path.insert(0, str(INTERVIEW_SRC))

import interview_environment as interview_env

importlib.reload(interview_env)

DEFAULT_SCENARIOS = list(interview_env.DEFAULT_SCENARIOS)
make_interview_conversation_record = interview_env.make_interview_conversation_record

SYSTEM_PROMPT = "You write realistic job-offer negotiation conversations. Output JSON only."
NEWLINE = chr(10)
PACKAGE_KEYWORDS = {
    "offer",
    "package",
    "comp",
    "compensation",
    "base",
    "sign-on",
    "sign on",
    "equity",
    "total comp",
}
CANDIDATE_NEGOTIATION_KEYWORDS = {
    "package",
    "offer",
    "comp",
    "compensation",
    "scope",
    "level",
    "flexibility",
    "growth",
    "weighing",
    "compare",
    "considering",
    "target",
    "closer",
    "move",
    "competitive",
    "room",
}
RECRUITER_NEGOTIATION_KEYWORDS = {
    "offer",
    "package",
    "comp",
    "compensation",
    "base",
    "sign-on",
    "sign on",
    "equity",
    "total comp",
    "scope",
    "role",
    "team",
    "growth",
    "flexibility",
    "band",
    "approved",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate interview conversation seed JSONL with vLLM."
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="vLLM / Hugging Face model name.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="JSONL output path for generated conversation seeds. Defaults to a model-specific path under Interview/Data/.",
    )
    parser.add_argument(
        "--dry_run_output_path",
        default=None,
        help="Optional single-record JSONL path used with --dry_run. Defaults to a model-specific *_dry_run.jsonl path under Interview/Data/.",
    )
    parser.add_argument("--total_conversations", type=int, default=5000)
    parser.add_argument("--turns_per_conversation", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_output_tokens", type=int, default=600)
    parser.add_argument("--request_pause_seconds", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional vLLM quantization mode, e.g. bitsandbytes. In this env, bitsandbytes defaults to 4-bit loading.",
    )
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument(
        "--run_tag",
        default=None,
        help="Optional run tag stored in metadata. Defaults to a model-specific tag.",
    )
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate a single example record and exit.",
    )
    parser.add_argument(
        "--print_example_prompt",
        action="store_true",
        help="Print the example prompt before generating.",
    )
    return parser.parse_args()


def slugify_name(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    return slug.strip("_").lower() or "model"


def default_output_stem(model_name: str, quantization: str | None = None) -> str:
    model_tail = str(model_name).split("/")[-1]
    stem = f"interview_conversation_seeds_vllm_{slugify_name(model_tail)}"
    if quantization:
        stem = f"{stem}_{slugify_name(quantization)}"
    return stem


def apply_default_paths_and_run_tag(args):
    if args.quantization is not None:
        args.quantization = str(args.quantization).strip() or None

    stem = default_output_stem(args.model_name, args.quantization)
    if not args.output_path:
        args.output_path = str(INTERVIEW_ROOT / "Data" / f"{stem}.jsonl")
    if not args.dry_run_output_path:
        args.dry_run_output_path = str(
            INTERVIEW_ROOT / "Data" / f"{stem}_dry_run.jsonl"
        )
    if not args.run_tag:
        args.run_tag = stem
    return args


def validate_args(args):
    if args.total_conversations <= 0:
        raise ValueError("--total_conversations must be positive.")
    if args.turns_per_conversation <= 0:
        raise ValueError("--turns_per_conversation must be positive.")
    if args.max_output_tokens <= 0:
        raise ValueError("--max_output_tokens must be positive.")
    if args.max_retries <= 0:
        raise ValueError("--max_retries must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log_every must be positive.")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards.")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor_parallel_size must be positive.")
    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        raise ValueError("--gpu_memory_utilization must be in (0, 1].")


def conversation_json_schema(turns_per_conversation: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "seeded_dialogue": {
                "type": "array",
                "minItems": int(turns_per_conversation),
                "maxItems": int(turns_per_conversation),
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["speaker", "message"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["seeded_dialogue"],
        "additionalProperties": False,
    }


def extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Model returned empty text.")

    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(candidate.strip() for candidate in fenced if candidate.strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ValueError("No valid JSON object found in model output." + NEWLINE + text)


def offer_total_comp(scenario) -> int:
    return (
        int(scenario.offer.base_salary)
        + int(scenario.offer.sign_on_bonus)
        + int(scenario.offer.equity_grant)
    )


def contains_any(text: str, keywords) -> bool:
    lowered = str(text).lower()
    return any(keyword in lowered for keyword in keywords)


def currency_mentions(text: str) -> int:
    return len(re.findall(r"\$\s?\d[\d,]*", str(text)))


def build_generation_prompt(scenario, conversation_id: str, turns_per_conversation: int) -> str:
    total_comp = offer_total_comp(scenario)
    return f"""
Write one realistic verbal-offer negotiation conversation variant.

Return JSON only in this shape:
{{
  "seeded_dialogue": [
    {{"speaker": "{scenario.interviewer_name}", "message": "..."}},
    {{"speaker": "Candidate", "message": "..."}}
  ]
}}

This should feel like a structured negotiation, not a generic follow-up chat.
It is the conversation right before the formal written offer email is sent.
The recruiter can verbally summarize the package, but should not say they are attaching or sending the written offer yet.

Required turn-by-turn flow:
1. Turn 1 from {scenario.interviewer_name}: verbally recap the role, the company, and the current package, then invite discussion.
2. Turn 2 from Candidate: respond positively but begin negotiating by surfacing concerns, tradeoffs, or pressure around the package, scope, growth, or flexibility.
3. Turn 3 from {scenario.interviewer_name}: respond like a recruiter in a negotiation by reinforcing value, clarifying details, or defending the package.
4. Turn 4 from Candidate: continue the negotiation naturally without making a final accept/reject decision.

Requirements:
- Exactly {turns_per_conversation} turns.
- Alternate speakers starting with {scenario.interviewer_name}, then Candidate, and continue alternating.
- Keep the dialogue professional, concise, and realistic.
- Each turn should be 1 to 3 sentences.
- The first recruiter turn must mention at least one concrete compensation detail and job details.
- The later turns should feel like an active negotiation over the package and role, not small talk.
- Do not include a final accept or reject decision.
- Do not invent a specific outside-offer salary.
- Make this variant feel materially different from obvious template phrasing.

Current package to anchor the negotiation:
- Company: {scenario.company_name}
- Role: {scenario.role_title}
- Team: {scenario.team_summary}
- Interviewer: {scenario.interviewer_name}
- Employer pitch: {scenario.employer_pitch}
- Base salary: ${scenario.offer.base_salary:,}
- Sign-on bonus: ${scenario.offer.sign_on_bonus:,}
- Equity grant: ${scenario.offer.equity_grant:,}
- Total comp on paper: ${total_comp:,}

Conversation ID:
- {conversation_id}
""".strip()


def validate_negotiation_shape(scenario, turns):
    first_turn = turns[0]
    first_message = str(first_turn["message"]).strip()
    if currency_mentions(first_message) < 1:
        raise ValueError(
            "First recruiter turn must mention at least one concrete dollar amount."
        )
    if not contains_any(first_message, PACKAGE_KEYWORDS):
        raise ValueError("First recruiter turn must mention the offer/package explicitly.")

    role_words = {
        word.lower() for word in str(scenario.role_title).split() if len(word) >= 4
    }
    role_words.add(str(scenario.company_name).lower())
    if not contains_any(first_message, role_words):
        raise ValueError("First recruiter turn must mention the company or role details.")

    candidate_followups = " ".join(
        str(turn["message"])
        for turn in turns[1:]
        if str(turn["speaker"]).strip() == "Candidate"
    )
    if not contains_any(candidate_followups, CANDIDATE_NEGOTIATION_KEYWORDS):
        raise ValueError("Candidate turns do not read like negotiation or pushback yet.")

    recruiter_followups = " ".join(
        str(turn["message"])
        for turn in turns[1:]
        if str(turn["speaker"]).strip() == str(scenario.interviewer_name)
    )
    if recruiter_followups and not contains_any(
        recruiter_followups, RECRUITER_NEGOTIATION_KEYWORDS
    ):
        raise ValueError(
            "Recruiter follow-up should continue the negotiation, not drift into generic pleasantries."
        )


def normalize_seeded_dialogue(scenario, raw_turns, turns_per_conversation: int):
    temp_record = make_interview_conversation_record(
        base_scenario_name=scenario.name,
        seeded_dialogue=raw_turns,
        conversation_id="validation_only",
    )
    turns = temp_record["seeded_dialogue"]
    if len(turns) != int(turns_per_conversation):
        raise ValueError(
            f"Expected {turns_per_conversation} turns, got {len(turns)}."
        )

    expected_speakers = [
        scenario.interviewer_name if idx % 2 == 0 else "Candidate"
        for idx in range(int(turns_per_conversation))
    ]
    for idx, (turn, expected_speaker) in enumerate(zip(turns, expected_speakers)):
        speaker = str(turn["speaker"]).strip()
        if speaker != expected_speaker:
            raise ValueError(
                f"Turn {idx} speaker mismatch. Expected {expected_speaker!r}, got {speaker!r}."
            )

    validate_negotiation_shape(scenario, turns)
    return turns


def build_generation_jobs(total_conversations: int, scenarios):
    jobs = []
    per_scenario_counts = Counter()
    scenario_list = list(scenarios)
    for global_idx in range(int(total_conversations)):
        scenario = scenario_list[global_idx % len(scenario_list)]
        per_scenario_idx = per_scenario_counts[scenario.name]
        per_scenario_counts[scenario.name] += 1
        conversation_id = f"{scenario.name}__{per_scenario_idx:05d}"
        jobs.append(
            {
                "global_idx": global_idx,
                "base_scenario_name": scenario.name,
                "scenario": scenario,
                "conversation_id": conversation_id,
            }
        )
    return jobs


def filter_jobs_for_shard(jobs, shard_index: int, num_shards: int):
    return [job for job in jobs if job["global_idx"] % num_shards == shard_index]


def append_jsonl(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj) + NEWLINE)


def read_existing_conversation_ids(path: Path):
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            conversation_id = row.get("conversation_id")
            if conversation_id:
                ids.add(str(conversation_id))
    return ids


def build_structured_outputs_params(turns_per_conversation: int) -> StructuredOutputsParams:
    return StructuredOutputsParams(
        json=conversation_json_schema(turns_per_conversation),
        disable_additional_properties=True,
    )


def make_llm(args) -> LLM:
    visible_cuda = torch.cuda.device_count()
    if visible_cuda < int(args.tensor_parallel_size):
        raise ValueError(
            f"Need at least {args.tensor_parallel_size} visible CUDA devices, found {visible_cuda}."
        )
    llm_kwargs = dict(
        model=args.model_name,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    return LLM(**llm_kwargs)


def call_vllm_raw(
    llm: LLM,
    prompt: str,
    sampling_seed: int,
    turns_per_conversation: int,
    args,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
        seed=int(sampling_seed),
        structured_outputs=build_structured_outputs_params(turns_per_conversation),
    )

    try:
        outputs = llm.chat(messages=[messages], sampling_params=[params], use_tqdm=False)
    except TypeError:
        outputs = llm.chat(messages=[messages], sampling_params=[params])

    if not outputs:
        raise RuntimeError("vLLM returned no outputs.")

    result = outputs[0]
    text = result.outputs[0].text if getattr(result, "outputs", None) else str(result)
    text = str(text).strip()
    if not text:
        raise RuntimeError("vLLM returned empty text.")
    return text


def generate_conversation_record(llm: LLM, job, args, attempt_seed: int = 0):
    scenario = job["scenario"]
    conversation_id = job["conversation_id"]
    prompt = build_generation_prompt(
        scenario,
        conversation_id=conversation_id,
        turns_per_conversation=args.turns_per_conversation,
    )
    raw_text = call_vllm_raw(
        llm,
        prompt,
        sampling_seed=args.seed
        + job["global_idx"] * 1000
        + args.shard_index
        + attempt_seed,
        turns_per_conversation=args.turns_per_conversation,
        args=args,
    )

    try:
        parsed = extract_json_object(raw_text)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse model output for {conversation_id}. Raw output follows:{NEWLINE}{raw_text}"
        ) from exc

    try:
        turns = normalize_seeded_dialogue(
            scenario,
            parsed.get("seeded_dialogue"),
            turns_per_conversation=args.turns_per_conversation,
        )
    except Exception as exc:
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        raise RuntimeError(
            f"Generated dialogue failed validation for {conversation_id}. Parsed output follows:{NEWLINE}{pretty}"
        ) from exc

    return make_interview_conversation_record(
        base_scenario_name=scenario.name,
        seeded_dialogue=turns,
        conversation_id=conversation_id,
        metadata={
            "run_tag": args.run_tag,
            "model_name": args.model_name,
            "quantization": args.quantization,
            "backend": "vllm",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_output_tokens": args.max_output_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "generated_at_unix": time.time(),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "global_idx": job["global_idx"],
            "attempt_seed": attempt_seed,
        },
    )


def generate_corpus(llm: LLM, jobs, output_path: Path, args) -> int:
    written = 0
    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        last_error = None
        for attempt in range(args.max_retries):
            try:
                record = generate_conversation_record(
                    llm,
                    job,
                    args,
                    attempt_seed=attempt,
                )
                append_jsonl(record, output_path)
                written += 1
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"[attempt {attempt + 1}/{args.max_retries}] failed for {job['conversation_id']}: {exc}",
                    flush=True,
                )
                time.sleep(min(2.0, 0.25 * (attempt + 1)))
        else:
            raise RuntimeError(
                f"Failed to generate conversation {job['conversation_id']} after {args.max_retries} tries"
            ) from last_error

        if args.request_pause_seconds > 0:
            time.sleep(args.request_pause_seconds)

        if idx % args.log_every == 0 or idx == total:
            print(
                f"generated {idx}/{total} new conversations -> {output_path}",
                flush=True,
            )
    return written


def choose_example_job(jobs, fallback_jobs):
    if jobs:
        return jobs[0]
    if fallback_jobs:
        return fallback_jobs[0]
    return None


def main():
    args = parse_args()
    args = apply_default_paths_and_run_tag(args)
    validate_args(args)

    output_path = Path(args.output_path)
    dry_run_output_path = Path(args.dry_run_output_path)

    all_jobs = build_generation_jobs(args.total_conversations, DEFAULT_SCENARIOS)
    shard_jobs = filter_jobs_for_shard(all_jobs, args.shard_index, args.num_shards)
    existing_ids = read_existing_conversation_ids(output_path) if args.resume else set()
    pending_jobs = [
        job for job in shard_jobs if str(job["conversation_id"]) not in existing_ids
    ]

    print(f"repo_root = {REPO_ROOT}", flush=True)
    print(f"cuda_device_count = {torch.cuda.device_count()}", flush=True)
    print(f"model = {args.model_name}", flush=True)
    print(f"quantization = {args.quantization}", flush=True)
    print(f"output_path = {output_path}", flush=True)
    print(f"total_conversations = {args.total_conversations}", flush=True)
    print(
        f"shard = {args.shard_index}/{args.num_shards} -> shard_jobs = {len(shard_jobs)}",
        flush=True,
    )
    print(f"existing_records = {len(existing_ids)}", flush=True)
    print(f"pending_jobs = {len(pending_jobs)}", flush=True)
    print(
        f"pending_by_scenario = {Counter(job['base_scenario_name'] for job in pending_jobs)}",
        flush=True,
    )

    example_job = choose_example_job(pending_jobs, shard_jobs)
    if example_job is None:
        print("No jobs available for this shard.", flush=True)
        return

    if args.print_example_prompt:
        example_prompt = build_generation_prompt(
            example_job["scenario"],
            conversation_id=example_job["conversation_id"],
            turns_per_conversation=args.turns_per_conversation,
        )
        print("=== EXAMPLE PROMPT ===", flush=True)
        print(example_prompt, flush=True)
        print(flush=True)

    if args.dry_run:
        llm = make_llm(args)
        record = generate_conversation_record(llm, example_job, args, attempt_seed=0)
        print("=== DRY RUN RECORD ===", flush=True)
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
        if dry_run_output_path:
            dry_run_output_path.parent.mkdir(parents=True, exist_ok=True)
            dry_run_output_path.write_text(
                json.dumps(record) + NEWLINE, encoding="utf-8"
            )
            print(f"dry_run_output_path = {dry_run_output_path}", flush=True)
        return

    if not pending_jobs:
        print("No pending jobs remaining for this shard.", flush=True)
        print("written_now = 0", flush=True)
        return

    llm = make_llm(args)
    written_now = generate_corpus(llm, pending_jobs, output_path, args)
    print(f"written_now = {written_now}", flush=True)


if __name__ == "__main__":
    main()
