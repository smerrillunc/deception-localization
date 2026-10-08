from __future__ import annotations

import json
import re
from typing import Any, Dict, List

try:
    from vllm import SamplingParams
except Exception:
    SamplingParams = None


class LLMAgent:
    def __init__(self, name, model, model_name, instruction_format: str = "reasoning", reasoning_instruction: str = "COT"):
        self.name = name
        self.model = model
        self.model_name = model_name
        self.hand: List[Any] = []
        self.reasoning_instruction = reasoning_instruction
        self.instruction_format = instruction_format

    def update_system_message(self, new_message):
        self.PLAYER_SYSTEM_MSG = new_message

    def update_BS_rules(self, new_rules):
        self.BS_RULES = new_rules

    def add_cards(self, cards):
        self.hand.extend(cards)

    def remove_cards(self, cards):
        for card in cards:
            if card in self.hand:
                self.hand.remove(card)

    def chat(
        self,
        messages,
        num_responses: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        debug: bool = False,
    ):
        if debug:
            print("=== Prompt ===")
            for msg in messages:
                print(f"Role: {msg['role']}\nContent: {msg['content']}\n{'-' * 40}")

        if SamplingParams is None:
            raise RuntimeError("vLLM is unavailable in this environment.")

        params_list = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=i,
            )
            for i in range(num_responses)
        ]

        prompt_batch = [messages for _ in range(num_responses)]
        results = self.model.chat(messages=prompt_batch, sampling_params=params_list)

        outputs = []
        for i, result in enumerate(results):
            raw_text = ""
            try:
                raw_text = result.outputs[0].text
            except Exception:
                raw_text = str(result)
            parsed = LLMAgent.parse_action(raw_text)
            parsed.update(
                {
                    "seed": i,
                    "prompt": messages[-1]["content"] if messages else "",
                }
            )
            outputs.append(parsed)

        if debug:
            for output in outputs:
                print(f"=== Seed {output['seed']} ===")
                print(output)
                print()

        return outputs

    @staticmethod
    def parse_action(raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        try:
            match = re.search(r"\{.*?\}", raw_text, flags=re.S)
            if not match:
                raise ValueError("No JSON object found")
            js_text = match.group()

            js_text = re.sub(r"//.*?(?=\n|$)", "", js_text)
            js_text = re.sub(r"/\*.*?\*/", "", js_text, flags=re.S)

            def remove_trailing_hash(line: str) -> str:
                in_str = False
                escaped = False
                for idx, ch in enumerate(line):
                    if ch == "\\" and not escaped:
                        escaped = True
                        continue
                    if ch in ('"', "'") and not escaped:
                        in_str = not in_str
                    if ch == "#" and not in_str:
                        return line[:idx].rstrip()
                    escaped = False
                return line

            js_text = "\n".join(remove_trailing_hash(line) for line in js_text.splitlines())
            js_text = re.sub(r",\s*}", "}", js_text)
            js_text = re.sub(r",\s*\]", "]", js_text)
            js_text = js_text.replace("\u201c", '"').replace("\u201d", '"')
            js_text = js_text.replace("\u2018", "'").replace("\u2019", "'")
            js_text = re.sub(r"\n+", " ", js_text)
            js_text = re.sub(r"(\w+)\s*:", r'"\1":', js_text)
            js_text = js_text.strip()
            return json.loads(js_text)
        except Exception as exc:
            print("COULD NOT PARSE JSON:", exc)
            print(raw_text)
            return {
                "Reasoning": raw_text,
                "Action": "PLAY",
                "Declared_Rank": None,
                "Card_idx": [],
                "Parse_fail": True,
            }
