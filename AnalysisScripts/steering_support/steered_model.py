"""Per-head activation steering on top of the transformers backend.

Steering is applied inside the generation backend rather than in a separate
script so that the steered and unsteered arms go through the same code path:
both are drawn at the same state by `deception_miner._model_actions` and
labelled by `deception_miner.deception_from_action`, so the two conditions
differ only by the intervention.

What it adds to the backend's `LLM`:

  * a forward **pre**-hook on each `self_attn.o_proj`, editing head h's slice of
    that module's input at the last position -- on the final prefill position
    and at every decode step;
  * a steering *window*, enforced by a LogitsProcessor that counts decode steps
    (and, for `reasoning:N`, watches for `</think>`) and switches the hooks off.

`set_steering(on)` toggles it, so one loaded model serves both conditions.

Delta modes, all acting on head h's `head_dim`-slice of the o_proj input:

    absolute     h += alpha * v
    relative     h += alpha * ||h|| * v_hat    (alpha is a fraction of the
                                                head's own activation scale)
    ablate       h -= alpha * (h . d_hat) d_hat
    ablate_add   ablate, then the relative add
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import torch

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import transformers_backend as hf_vllm_shim  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

THINK_CLOSE = "</think>"
# `</think>` is a single token for this tokenizer (id 151649), so a sequence can
# be marked closed the step it emits one, exactly and without decoding.
THINK_CLOSE_ID = 151649


def load_vectors(path: Path) -> dict[tuple[int, int], torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    raw = payload["vectors"] if isinstance(payload, dict) and "vectors" in payload else payload
    out: dict[tuple[int, int], torch.Tensor] = {}
    for key, value in dict(raw).items():
        if isinstance(key, tuple):
            site = (int(key[0]), int(key[1]))
        else:
            m = re.match(r"layer_(\d+)_head_(\d+)", str(key))
            if not m:
                raise ValueError(f"bad vector key {key!r}")
            site = (int(m.group(1)), int(m.group(2)))
        out[site] = value.detach().float().cpu()
    return out


def load_site_scale(path: Path) -> dict[tuple[int, int], float]:
    """Optional per-site dose multiplier stored alongside the vectors.

    Every dose sweep so far spends the same alpha on the highest-attribution
    head and the 128th one. If attribution is causal that is the wrong budget
    split: the per-head cliff near alpha=0.32 caps what any single site can
    take, so the way to buy more reduction at equal total dose is to move dose
    from sites that barely matter onto sites that do. A fold file may carry
    `site_scale`; absent, every site weighs 1.0 and behaviour is unchanged.
    """
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return {}
    raw = (payload.get("metadata") or {}).get("site_scale") or payload.get("site_scale")
    if not raw:
        return {}
    out: dict[tuple[int, int], float] = {}
    for key, value in dict(raw).items():
        if isinstance(key, tuple):
            site = (int(key[0]), int(key[1]))
        else:
            m = re.match(r"layer_(\d+)_head_(\d+)", str(key))
            if not m:
                continue
            site = (int(m.group(1)), int(m.group(2)))
        out[site] = float(value)
    return out


class _WindowProcessor(LogitsProcessor):
    """Counts decode steps and closes the steering window.

    HF calls this once per generation step with the running `input_ids`, which is
    exactly the hook we need: there is no per-token callback on `generate`, but a
    logits processor is called at every step and may have side effects.
    """

    def __init__(self, state: dict[str, Any], *, kind: str, budget: int | None,
                 prompt_len: int, tokenizer, start: int = 0) -> None:
        self.state, self.kind, self.budget = state, kind, budget
        self.prompt_len, self.tokenizer, self.start = prompt_len, tokenizer, start

    def __call__(self, input_ids, scores):
        n_new = int(input_ids.shape[1]) - self.prompt_len
        if self.kind == "delayed":
            # Off until `start`, then on for `budget` tokens.  This is the lever for
            # applying the circuit where it was actually fit -- part-way into a
            # reasoning trace -- rather than at the first token of the turn, where
            # no reasoning exists yet.
            if n_new < self.start:
                self.state["on"] = False
                self.state["stop_reason"] = "not_started"
            elif self.budget is None or n_new < self.start + self.budget:
                self.state["on"] = True
                self.state["stop_reason"] = "active"
                self.state["n_steered"] = n_new - self.start
            else:
                self.state["on"] = False
                self.state["stop_reason"] = "token_budget"
            return scores
        if not self.state.get("on"):
            return scores
        self.state["n_steered"] = n_new
        if self.kind == "whole":
            return scores
        if self.budget is not None and n_new >= self.budget:
            self.state["on"] = False
            self.state["stop_reason"] = "token_budget"
            return scores
        if self.kind == "reasoning":
            # PER-SEQUENCE gating. Generation is batched, so sequences reach
            # `</think>` at different steps. Closing the window only when every
            # row has closed would keep steering the rows that finished early --
            # straight through the action JSON, which is the thing this window
            # exists to avoid. Instead each row is latched off at its own
            # `</think>` and the hook applies the delta only to rows still
            # inside their reasoning.
            gen = input_ids[:, self.prompt_len:]
            closed = self.state.get("row_closed")
            if closed is None or closed.shape[0] != gen.shape[0]:
                closed = torch.zeros(gen.shape[0], dtype=torch.bool,
                                     device=input_ids.device)
            if gen.shape[1] > 0:
                closed = closed | (gen[:, -1] == THINK_CLOSE_ID)
            self.state["row_closed"] = closed
            self.state["row_on"] = ~closed
            n_open = int((~closed).sum())
            if n_open == 0:
                self.state["on"] = False
                self.state["stop_reason"] = "think_closed"
            else:
                self.state["stop_reason"] = "active"
            self.state["n_rows_open"] = n_open
            self.state["max_rows_closed"] = max(
                int(self.state.get("max_rows_closed", 0)), int(closed.sum()))
            self.state["proc_calls"] = int(self.state.get("proc_calls", 0)) + 1
        return scores


class SteeredLLM(hf_vllm_shim.LLM):
    def __init__(self, *args: Any, vector_path: str, alpha: float,
                 window: str = "reasoning:400", delta_mode: str = "absolute",
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vectors = load_vectors(Path(vector_path))
        self.site_scale = load_site_scale(Path(vector_path))
        self.alpha = float(alpha)
        self.window = str(window)
        self.delta_mode = str(delta_mode)
        cfg = self.model.config
        n_heads = int(cfg.num_attention_heads)
        self.head_dim = int(getattr(cfg, "head_dim", 0) or (cfg.hidden_size // n_heads))
        self._state: dict[str, Any] = {"on": False, "n_steered": 0, "stop_reason": "not_applied"}
        self._steer_enabled = False
        if self.delta_mode == "resid":
            self._install_resid_hooks()
        else:
            self._install_hooks()

    # -- window spec ------------------------------------------------------- #
    def _window_parts(self) -> tuple[str, int | None, int]:
        spec = self.window.strip().lower()
        if spec in {"whole", "all"}:
            return "whole", None, 0
        m = re.fullmatch(r"delayed:(\d+):(\d+)", spec)
        if m:
            return "delayed", int(m.group(2)), int(m.group(1))
        m = re.fullmatch(r"(fixed|reasoning|prejson):(\d+)", spec)
        if not m:
            raise ValueError(f"bad window {self.window!r}")
        return m.group(1), int(m.group(2)), 0

    # -- hooks ------------------------------------------------------------- #
    def _install_hooks(self) -> None:
        """Install the per-head intervention.

        Four modes, all acting on head h's 128-dim slice of the o_proj input:

          absolute    h += alpha * v
          relative    h += alpha * ||h|| * v_hat        (alpha is a fraction of
                      the head's own activation, so every site gets the same
                      dose -- absolute alpha ranged 0.73x-1.90x across sites)
          ablate      h -= alpha * (h . d_hat) d_hat    where d_hat is the unit
                      DECEPTION direction, i.e. -v_hat. This removes the
                      component the model already has along deception rather
                      than pushing along truthfulness. alpha=1 removes it
                      entirely. Unlike the additive modes it cannot overshoot:
                      the result is a projection, bounded by ||h||, which is
                      what has capped every additive dose sweep so far.
          ablate_add  ablate, then add back along truthfulness -- both halves at
                      the same alpha.
        """
        by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in self.vectors:
            by_layer.setdefault(int(layer_idx), []).append(int(head_idx))
        device = next(self.model.parameters()).device
        mode = self.delta_mode
        for layer_idx, heads in by_layer.items():
            module = self.model.model.layers[layer_idx].self_attn.o_proj
            specs = []
            for h in sorted(heads):
                vec = self.vectors[(layer_idx, h)].to(device)
                norm = float(vec.norm())
                unit = (vec / norm) if norm > 0 else vec
                # stored vec is mean(truthful) - mean(deceptive); the deception
                # direction is its negation
                w = float(self.site_scale.get((layer_idx, h), 1.0))
                specs.append((h * self.head_dim, vec, unit, (-unit), w))

            def pre_hook(_mod, args, specs=specs, mode=mode, alpha=self.alpha):
                if not self._state.get("on") or not args or not torch.is_tensor(args[0]):
                    return None
                hidden = args[0]
                patched = hidden.clone()
                # rows whose reasoning has already closed are left untouched
                row_on = self._state.get("row_on")
                if row_on is not None:
                    if row_on.shape[0] != hidden.shape[0] or not bool(row_on.any()):
                        if row_on.shape[0] == hidden.shape[0] and not bool(row_on.any()):
                            return None
                        row_on = None
                mask = (row_on.to(hidden.dtype).view(-1, 1, 1)
                        if row_on is not None else None)
                st = self._state
                st["hook_calls"] = int(st.get("hook_calls", 0)) + 1
                if mask is not None and not bool(row_on.all()):
                    st["masked_calls"] = int(st.get("masked_calls", 0)) + 1
                elif self._state.get("row_on") is not None and mask is None:
                    st["shape_mismatch"] = int(st.get("shape_mismatch", 0)) + 1
                for off, vec, unit, dec, w in specs:
                    width = vec.shape[-1]
                    a = alpha * w
                    block = patched[:, -1:, off : off + width]
                    if mode == "absolute":
                        d = (a * vec).to(hidden.dtype).view(1, 1, -1).expand_as(block)
                        block += d if mask is None else d * mask
                    elif mode == "relative":
                        d = ((a * unit).to(hidden.dtype).view(1, 1, -1)
                             * block.norm(dim=-1, keepdim=True))
                        block += d if mask is None else d * mask
                    elif mode in ("ablate", "ablate_add"):
                        dh = dec.to(hidden.dtype).view(1, 1, -1)
                        comp = (block * dh).sum(dim=-1, keepdim=True)   # h . d_hat
                        block -= a * comp * dh
                        if mode == "ablate_add":
                            block += ((a * unit).to(hidden.dtype).view(1, 1, -1)
                                      * block.norm(dim=-1, keepdim=True))
                    else:
                        raise ValueError(f"bad delta_mode {mode!r}")
                new_args = list(args)
                new_args[0] = patched
                return tuple(new_args)

            module.register_forward_pre_hook(pre_hook)

    def _install_resid_hooks(self) -> None:
        """Steer the RESIDUAL STREAM instead of individual head slices.

        Per-head steering edits a 128-dim slice of one layer's o_proj input. The
        classic activation-steering formulation instead adds a direction to the
        whole residual stream at a layer, which reaches every downstream head
        and MLP rather than one attention path. Per-head edits have capped out
        around -0.24 here; this is a coarser but much broader intervention.

        The layer direction is the concatenation of that layer's head vectors
        placed in their own slices (zeros elsewhere), so it is built from the
        same attribution, not a new quantity. alpha is relative to the residual
        norm, as in `relative` mode.
        """
        by_layer: dict[int, dict[int, torch.Tensor]] = {}
        for (layer_idx, head_idx), vec in self.vectors.items():
            by_layer.setdefault(int(layer_idx), {})[int(head_idx)] = vec
        device = next(self.model.parameters()).device
        hidden_size = int(self.model.config.hidden_size)
        for layer_idx, heads in by_layer.items():
            full = torch.zeros(hidden_size, device=device)
            for h, vec in heads.items():
                full[h * self.head_dim:(h + 1) * self.head_dim] = vec.to(device)
            n = float(full.norm())
            if n > 0:
                full = full / n
            layer = self.model.model.layers[layer_idx]

            def post_hook(_mod, _args, output, full=full, alpha=self.alpha):
                if not self._state.get("on"):
                    return output
                hs = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(hs):
                    return output
                patched = hs.clone()
                blk = patched[:, -1:, :]
                blk += (alpha * full).to(hs.dtype).view(1, 1, -1) * blk.norm(dim=-1, keepdim=True)
                if isinstance(output, tuple):
                    return (patched,) + tuple(output[1:])
                return patched

            layer.register_forward_hook(post_hook)

    def set_steering(self, on: bool) -> None:
        self._steer_enabled = bool(on)

    def last_steering_info(self) -> dict[str, Any]:
        return {
            "steering_tokens": int(self._state.get("n_steered", 0)),
            "steering_stop_reason": self._state.get("stop_reason", "not_applied"),
        }

    # -- generation -------------------------------------------------------- #
    @torch.no_grad()
    def chat(self, messages: Any, sampling_params: Any = None, **kwargs: Any):
        if not self._steer_enabled:
            self._state.update(on=False, n_steered=0, stop_reason="not_applied",
                               row_closed=None, row_on=None)
            return super().chat(messages, sampling_params=sampling_params, **kwargs)

        kind, budget, start = self._window_parts()
        # a delayed window must not steer the prefill
        # the per-row think-block latch is per generation; carrying it over would
        # leave a later batch steering nothing, or the wrong rows
        self._state.update(row_closed=None, row_on=None)
        self._state.update(on=(kind != "delayed"), n_steered=0,
                           stop_reason="not_started" if kind == "delayed" else "active")

        orig_generate = self.model.generate

        def generate_with_window(**enc_and_kwargs):
            prompt_len = int(enc_and_kwargs["input_ids"].shape[1])
            proc = _WindowProcessor(self._state, kind=kind, budget=budget,
                                    prompt_len=prompt_len, tokenizer=self.tokenizer,
                                    start=start)
            existing = enc_and_kwargs.pop("logits_processor", None)
            lp = LogitsProcessorList(list(existing) if existing else [])
            lp.append(proc)
            return orig_generate(logits_processor=lp, **enc_and_kwargs)

        self.model.generate = generate_with_window
        try:
            out = super().chat(messages, sampling_params=sampling_params, **kwargs)
        finally:
            self.model.generate = orig_generate
            if self._state.get("stop_reason") == "active":
                self._state["stop_reason"] = "generation_ended"
            self._state["on"] = False
        return out
