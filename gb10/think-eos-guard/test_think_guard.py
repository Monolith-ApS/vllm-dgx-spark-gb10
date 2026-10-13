"""Offline tests for think_guard.py (no vLLM, no GPU): a scripted fake engine stands in for AsyncLLM.generate().
Run: python test_think_guard.py   (from this directory)."""
import asyncio
import enum
import importlib.util
import logging
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

# --- minimal stand-ins for the two vllm modules think_guard imports --------------------------------------------------
vllm = types.ModuleType("vllm")
vllm_logger = types.ModuleType("vllm.logger")
vllm_logger.init_logger = lambda name: logging.getLogger(name)
vllm_sp = types.ModuleType("vllm.sampling_params")


class RequestOutputKind(enum.Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


vllm_sp.RequestOutputKind = RequestOutputKind
sys.modules.update({"vllm": vllm, "vllm.logger": vllm_logger, "vllm.sampling_params": vllm_sp})
spec = importlib.util.spec_from_file_location("think_guard", os.path.join(HERE, "think_guard.py"))
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)

END, OPEN, TOOLT, IM_END, EOT, NL = 100, 101, 102, 103, 104, 10
VOCAB = {"</think>": END, "<think>": OPEN, "<tool_call>": TOOLT, "<|im_end|>": IM_END, "<|endoftext|>": EOT}
SAFE = [201, 202, 203]  # "</", ZWSP, "think>"


class Tok:
    eos_token_id = IM_END
    unk_token_id = 0

    def convert_tokens_to_ids(self, t):
        return VOCAB.get(t, 0)

    def encode(self, text, add_special_tokens=False):
        return list(SAFE)

    def decode(self, ids):
        return tg.SAFE_TAG if list(ids) == SAFE else "?"


class SP:
    def __init__(self, max_tokens=1000, kind=RequestOutputKind.DELTA):
        self.max_tokens, self.min_tokens, self.n, self.logprobs, self.structured_outputs, self.output_kind = max_tokens, 0, 1, None, None, kind

    def clone(self):
        c = SP(self.max_tokens, self.output_kind)
        c.min_tokens = self.min_tokens
        return c


class Out:
    def __init__(self, text, ids, finish=None):
        self.index, self.text, self.token_ids, self.finish_reason, self.stop_reason = 0, text, list(ids), finish, None


class Res:
    def __init__(self, prompt_ids, out):
        self.request_id, self.prompt_token_ids, self.outputs, self.finished = "r", list(prompt_ids), [out], out.finish_reason is not None


class Engine:
    """script: one list of (text, ids, finish) steps per generate() call."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate(self, engine_input, params, request_id, **kw):
        self.calls.append({"prompt": list(engine_input["prompt_token_ids"]), "params": params, "id": request_id, "kw": kw})
        steps = self.script.pop(0)
        prompt = list(engine_input["prompt_token_ids"])

        async def gen():
            for text, ids, fin in steps:
                yield Res(prompt, Out(text, ids, fin))
        return gen()


PROMPT = [5, 6, OPEN, NL]  # a rendered prompt that ends with an open <think>


def run(engine, params=None, prompt=None, request=None):
    params = params or SP()
    inp = {"type": "token", "prompt_token_ids": list(prompt or PROMPT)}
    g = tg.wrap_generate(engine.generate, inp, params, "rid", {"session_id": "s"}, Tok(), request or types.SimpleNamespace(n=1))

    async def collect():
        return [r async for r in g]
    return asyncio.run(collect())


def text_of(rs):
    return "".join(r.outputs[0].text for r in rs)


def ids_of(rs):
    return [i for r in rs for i in r.outputs[0].token_ids]


class GuardTests(unittest.TestCase):
    def test_genuine_close_passes_through_untouched(self):
        e = Engine([[("plan", [1, 2], None), ("\n", [NL], None), ("</think>", [END], None), ("\n\nOK", [3, 4, IM_END], "stop")]])
        rs = run(e)
        self.assertEqual(text_of(rs), "plan\n</think>\n\nOK")
        self.assertEqual(ids_of(rs), [1, 2, NL, END, 3, 4, IM_END])
        self.assertEqual(len(e.calls), 1)
        self.assertEqual(rs[-1].outputs[0].finish_reason, "stop")

    def test_eos_inside_reasoning_is_resumed_as_one_stream(self):
        e = Engine([[("tag (`", [1, 2], None), ("", [IM_END], "stop")],
                    [("quoted", [7], None), ("\n</think>\n\nDone", [NL, END, 8, IM_END], "stop")]])
        rs = run(e)
        self.assertEqual(len(e.calls), 2)
        self.assertEqual(e.calls[1]["prompt"], PROMPT + [1, 2])  # EOS removed
        self.assertEqual(e.calls[1]["params"].min_tokens, 12)
        self.assertEqual(e.calls[1]["params"].max_tokens, 998)
        self.assertEqual(e.calls[1]["id"], "rid-r1")
        self.assertEqual(e.calls[1]["kw"], {"session_id": "s"})
        self.assertEqual(text_of(rs), "tag (`quoted\n</think>\n\nDone")
        self.assertEqual([r.outputs[0].finish_reason for r in rs][:-1], [None] * (len(rs) - 1))
        self.assertEqual(rs[-1].outputs[0].finish_reason, "stop")
        self.assertTrue(all(r.prompt_token_ids == PROMPT for r in rs))

    def test_quoted_tag_is_rewritten_and_does_not_close(self):
        e = Engine([[('marker = "', [1], None), ("</think>", [END], None), ('"', [2], None), ("", [IM_END], "stop")],
                    [("\n</think>\n\nA", [NL, END, 9, IM_END], "stop")]])
        rs = run(e)
        self.assertEqual(text_of(rs), 'marker = "' + tg.SAFE_TAG + '"\n</think>\n\nA')
        self.assertNotIn(END, ids_of(rs)[:5])
        self.assertEqual(ids_of(rs)[1:4], SAFE)
        self.assertEqual(len(e.calls), 2)  # the quote did not count as a close, so the EOS after it was resumed
        self.assertEqual(e.calls[1]["prompt"], PROMPT + [1, END, 2])  # the model's own tokens, not the rewrite

    def test_tag_after_the_genuine_close_is_kept_visible(self):
        e = Engine([[("x\n", [1, NL], None), ("</think>", [END], None), ("\n\nHe wrote ", [3], None), ("</think>", [END], None), ("!", [IM_END], "stop")]])
        rs = run(e)
        self.assertEqual(text_of(rs), "x\n</think>\n\nHe wrote " + tg.SAFE_TAG + "!")
        self.assertEqual(ids_of(rs).count(END), 1)

    def test_tool_call_ends_the_turn_even_without_a_close(self):
        e = Engine([[("think", [1], None), ("<tool_call>\n<function=f>", [TOOLT, 2], None), ("", [IM_END], "stop")]])
        rs = run(e)
        self.assertEqual(len(e.calls), 1)
        self.assertEqual(rs[-1].outputs[0].finish_reason, "stop")

    def test_thinking_closed_in_prompt_is_not_guarded(self):
        e = Engine([[("hi", [1, IM_END], "stop")]])
        rs = run(e, prompt=[5, OPEN, NL, END, NL, NL])
        self.assertEqual(len(rs), 1)
        self.assertEqual(len(e.calls), 1)

    def test_resume_budget_is_limited(self):
        os.environ["THINK_GUARD_MAX_RESUMES"] = "2"
        try:
            e = Engine([[("a", [1, IM_END], "stop")], [("b", [2, IM_END], "stop")], [("c", [3, IM_END], "stop")]])
            rs = run(e)
        finally:
            del os.environ["THINK_GUARD_MAX_RESUMES"]
        self.assertEqual(len(e.calls), 3)
        self.assertEqual(rs[-1].outputs[0].finish_reason, "stop")
        self.assertEqual(text_of(rs), "abc")

    def test_max_tokens_is_respected(self):
        e = Engine([[("a", [1, 2, 3, IM_END], "stop")]])
        rs = run(e, params=SP(max_tokens=3))  # nothing left to continue with
        self.assertEqual(len(e.calls), 1)
        self.assertEqual(rs[-1].outputs[0].finish_reason, "stop")

    def test_final_only_merges_segments(self):
        e = Engine([[("tag (`", [1, 2, IM_END], "stop")], [("rest\n</think>\n\nOK", [3, NL, END, 4, IM_END], "stop")]])
        rs = run(e, params=SP(kind=RequestOutputKind.FINAL_ONLY))
        self.assertEqual(len(rs), 1)
        self.assertEqual(rs[0].outputs[0].text, "tag (`rest\n</think>\n\nOK")
        self.assertEqual(rs[0].outputs[0].token_ids, [1, 2, 3, NL, END, 4, IM_END])
        self.assertEqual(rs[0].prompt_token_ids, PROMPT)

    def test_kill_switch_and_ineligible_requests(self):
        os.environ["THINK_GUARD"] = "0"
        try:
            e = Engine([[("a", [1, IM_END], "stop")]])
            self.assertEqual(len(run(e)), 1)
            self.assertEqual(len(e.calls), 1)
        finally:
            del os.environ["THINK_GUARD"]
        e = Engine([[("a", [1, IM_END], "stop")]])
        run(e, request=types.SimpleNamespace(n=2))
        self.assertEqual(len(e.calls), 1)

    def test_text_and_ids_out_of_step_fails_open(self):
        e = Engine([[("x</think>", [1, END, END], "stop")]])  # two ids, one tag in the text
        rs = run(e)
        self.assertEqual(text_of(rs), "x</think>")


if __name__ == "__main__":
    unittest.main()
