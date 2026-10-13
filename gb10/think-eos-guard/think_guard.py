# SPDX-License-Identifier: Apache-2.0
"""Think guard: keep a Qwen3.8 turn alive when it samples EOS inside an unfinished reasoning block, and keep a quoted closing
think tag from ending the reasoning.

Observed on a DGX Spark: while Qwen3.8 Flash Next reasons about text that contains the
closing think tag, it samples <|im_end|> (EOS) where it was about to quote the tag, in about half of the raw runs we
measured, or it writes the tag token itself as a quoted string. Either way vLLM returns finish_reason=stop with the whole
output in the reasoning channel (empty content, no tool call), or the stock parser ends the reasoning at the quote. An
agent harness reads that as a finished turn (upstream vLLM #55420, Qwen3.8 #216).

This module wraps AsyncLLM.generate() for chat requests (called from serving.py):

* Resume: when the engine finishes with EOS while the reasoning block is still open (no genuine closing tag, no tool call),
  the generation is resumed from prompt + what was generated (EOS removed), up to THINK_GUARD_MAX_RESUMES times, with
  min_tokens=THINK_GUARD_MIN_TOKENS so EOS cannot be sampled again straight away. The client sees one uninterrupted stream.
* Quoted tag: the closing tag token counts as a genuine end of reasoning only when it directly follows a newline (the
  template and the model write "...\n</think>\n\n"). Any other occurrence, and any after the genuine one, is rewritten to
  "</" + ZERO WIDTH SPACE + "think>" before the stock parser sees it, so it stays visible text instead of ending the
  reasoning or being swallowed from the answer.

Sampling is not touched, so speculative decoding (MTP) keeps working. THINK_GUARD=0 turns everything off.
"""
from __future__ import annotations

import os
import re
from collections.abc import AsyncGenerator, Callable
from typing import Any

from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

logger = init_logger(__name__)

TAG = "</think>"
OPEN = "<think>"
TOOL = "<tool_call>"
SAFE_TAG = "</​think>"
EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")
_TAG_RE = re.compile(re.escape(TAG))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def guard_enabled() -> bool:
    return os.environ.get("THINK_GUARD", "1") != "0"


class Guard:
    """Per-request state. Built from the tokenizer and the rendered prompt; `active` is False when the guard does not apply
    (thinking closed in the prompt, or the vocabulary lacks the tags)."""

    def __init__(self, tokenizer: Any, prompt_ids: list[int]):
        self.active = False
        try:
            self.end_id = tokenizer.convert_tokens_to_ids(TAG)
            self.open_id = tokenizer.convert_tokens_to_ids(OPEN)
            unk = getattr(tokenizer, "unk_token_id", None)
            eos = {tokenizer.convert_tokens_to_ids(t) for t in EOS_TOKENS}
            eos.add(getattr(tokenizer, "eos_token_id", None))
            self.eos_ids = {i for i in eos if isinstance(i, int) and i >= 0 and i != unk}
            self.safe_ids = list(tokenizer.encode(SAFE_TAG, add_special_tokens=False))
            if (not isinstance(self.end_id, int) or not isinstance(self.open_id, int) or self.end_id == unk
                    or self.open_id == unk or not self.eos_ids or self.end_id in self.safe_ids
                    or tokenizer.decode(self.safe_ids) != SAFE_TAG):
                return
        except Exception:  # an unfamiliar tokenizer: stay out of the way
            return
        last_open = max((i for i, t in enumerate(prompt_ids) if t == self.open_id), default=-1)
        last_end = max((i for i, t in enumerate(prompt_ids) if t == self.end_id), default=-1)
        self.active = last_open > last_end  # the template ended the prompt with an open <think>
        self.closed = False  # a genuine closing tag was generated
        self.tool_seen = False
        self.tail = "\n"  # the prompt ends "<think>\n"
        self.rewritten = 0

    def sanitize(self, text: str, ids: list[int]) -> tuple[str, list[int]]:
        """Rewrite quoted closing tags in one output delta; track the genuine close and tool calls."""
        if not text and not ids:
            return text, ids
        has_id = self.end_id in ids
        if not has_id and TAG not in text:
            if TOOL in text:
                self.tool_seen = True
            if text:
                self.tail = text[-1]
            return text, ids
        pos = [m.start() for m in _TAG_RE.finditer(text)]
        idpos = [k for k, t in enumerate(ids) if t == self.end_id]
        if len(pos) != len(idpos):  # text and ids are out of step (partial UTF-8 at a delta edge): leave it to the stock parser
            self.closed = True
            self.tail = text[-1] if text else self.tail
            return text, ids
        out_text: list[str] = []
        out_ids: list[int] = []
        t_last = 0
        i_last = 0
        for p, k in zip(pos, idpos):
            before = text[p - 1] if p > 0 else self.tail
            genuine = before == "\n" and not self.closed
            out_text.append(text[t_last:p])
            out_ids.extend(ids[i_last:k])
            if genuine:
                self.closed = True
                out_text.append(TAG)
                out_ids.append(self.end_id)
            else:
                self.rewritten += 1
                out_text.append(SAFE_TAG)
                out_ids.extend(self.safe_ids)
            t_last = p + len(TAG)
            i_last = k + 1
        out_text.append(text[t_last:])
        out_ids.extend(ids[i_last:])
        new_text = "".join(out_text)
        if TOOL in new_text:
            self.tool_seen = True
        self.tail = text[-1] if text else self.tail
        return new_text, out_ids

    def premature(self, raw_ids: list[int]) -> bool:
        """True when the engine ended on EOS while the reasoning block is still open."""
        return bool(self.active and not self.closed and not self.tool_seen and raw_ids and raw_ids[-1] in self.eos_ids)


def applicable(sampling_params: Any, engine_input: Any, request: Any) -> bool:
    if not guard_enabled():
        return False
    if getattr(request, "n", None) not in (None, 1) or getattr(request, "use_beam_search", False):
        return False
    if not isinstance(engine_input, dict) or engine_input.get("type") != "token" or not engine_input.get("prompt_token_ids"):
        return False
    sp = sampling_params
    if getattr(sp, "n", 1) != 1 or getattr(sp, "logprobs", None) is not None or getattr(sp, "structured_outputs", None):
        return False
    return getattr(sp, "output_kind", None) in (RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY)


def wrap_generate(generate: Callable[..., AsyncGenerator], engine_input: dict, sampling_params: Any, request_id: str,
                  gen_kwargs: dict, tokenizer: Any, request: Any):
    """The generator serving.py iterates: the plain engine stream, or the guarded one when the guard applies."""
    if not applicable(sampling_params, engine_input, request):
        return generate(engine_input, sampling_params, request_id, **gen_kwargs)
    guard = Guard(tokenizer, list(engine_input["prompt_token_ids"]))
    if not guard.active:
        return generate(engine_input, sampling_params, request_id, **gen_kwargs)
    return _guarded(generate, engine_input, sampling_params, request_id, gen_kwargs, guard)


async def _guarded(generate, engine_input, sampling_params, request_id, gen_kwargs, g: Guard):
    final_only = sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    max_resumes = _env_int("THINK_GUARD_MAX_RESUMES", 6)
    min_tokens = _env_int("THINK_GUARD_MIN_TOKENS", 12)
    orig_prompt = list(engine_input["prompt_token_ids"])
    raw_ids: list[int] = []  # everything the engine generated, EOS removed: the continuation prompt is orig_prompt + raw_ids
    pre_text = ""  # FINAL_ONLY: sanitized text and ids of earlier segments
    pre_ids: list[int] = []
    cur_input, cur_params, cur_id = engine_input, sampling_params, request_id
    attempt = 0
    while True:
        resume = False
        inner = generate(cur_input, cur_params, cur_id, **gen_kwargs)
        try:
            async for res in inner:
                res.prompt_token_ids = orig_prompt
                out = res.outputs[0]
                seg_ids = list(out.token_ids)
                text, ids = g.sanitize(out.text, seg_ids)
                if out.finish_reason is not None:
                    seg_raw = raw_ids + seg_ids
                    if attempt < max_resumes and g.premature(seg_raw):
                        # not the end: forward what we have, without finish, and continue
                        while seg_ids and seg_ids[-1] in g.eos_ids:
                            seg_ids.pop()
                            if ids and ids[-1] in g.eos_ids:
                                ids = ids[:-1]
                        raw_ids = raw_ids + seg_ids
                        remaining = None if sampling_params.max_tokens is None else sampling_params.max_tokens - len(raw_ids)
                        if remaining is None or remaining > 0:
                            resume = True
                            if final_only:
                                pre_text += text
                                pre_ids += ids
                            else:
                                out.text, out.token_ids = text, ids
                                out.finish_reason, out.stop_reason = None, None
                                res.finished = False
                                if text or ids:
                                    yield res
                            break
                    # genuine end of this turn (or nothing left to resume)
                    if final_only:
                        out.text, out.token_ids = pre_text + text, pre_ids + ids
                    else:
                        out.text, out.token_ids = text, ids
                    if attempt or g.rewritten:
                        logger.info("think-guard %s: finished after %d resume(s), %d quoted closing tag(s) rewritten", request_id, attempt, g.rewritten)
                    yield res
                    return
                raw_ids.extend(seg_ids)
                out.text, out.token_ids = text, ids
                yield res
        finally:
            await inner.aclose()
        if not resume:
            return
        attempt += 1
        logger.warning("think-guard %s: EOS inside open reasoning after %d generated tokens, resuming (attempt %d/%d)", request_id, len(raw_ids), attempt, max_resumes)
        cur_input = {"type": "token", "prompt_token_ids": orig_prompt + raw_ids}
        if engine_input.get("cache_salt") is not None:
            cur_input["cache_salt"] = engine_input["cache_salt"]
        cur_params = sampling_params.clone()
        if sampling_params.max_tokens is not None:
            cur_params.max_tokens = sampling_params.max_tokens - len(raw_ids)
        cur_params.min_tokens = max(sampling_params.min_tokens, min(min_tokens, cur_params.max_tokens or min_tokens))
        cur_id = "%s-r%d" % (request_id, attempt)
