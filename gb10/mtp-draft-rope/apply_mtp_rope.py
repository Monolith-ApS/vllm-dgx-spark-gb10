#!/usr/bin/env python3
"""vLLM 0.32: give the target's own MTP head the target's dict hf_overrides.

  python3 apply_mtp_rope.py <site-packages>/vllm

Our long-context profiles add YaRN through a dict --hf-overrides ({"text_config": {"rope_parameters": {... "rope_type":
"yarn", "factor": F, "original_max_position_embeddings": 262144}}}). SpeculativeConfig applies only callable overrides to the
draft ("Dict overrides stay target-only"), so the MTP draft builds plain MRoPE from config.json (max_position_embeddings
262,144). vLLM 0.32 sizes the MRoPE cos/sin cache as max_position_embeddings * mrope_positions_factor(), and that factor is
1 for a model without video pruning (0.31 hard-coded 4). The draft's table therefore ends at position 262,144 and the first
decode step past it reads outside the table: "CUDA error: an illegal memory access" at ~264.9K tokens on every profile above
256K (fill test 2026-10-23, 8 of 8 runs, marlin and humming, with and without the warmup fix).

With method "mtp" and a draft that is the target checkpoint itself, the draft now gets the target's dict overrides before its
MTP conversion, so its rope is the same YaRN as the target's and covers max_model_len. Any other draft keeps the stock rule.
Refuses an anchor that does not match exactly once.
"""
import os
import sys

pkg = sys.argv[1]
p = os.path.join(pkg, "config/speculative.py")
s = open(p, encoding="utf-8").read()

pairs = [(
    """    @staticmethod
    def _apply_composed_hf_override(
""",
    """    @staticmethod
    def _gb10_apply_target_dict(target_overrides: dict, hf_config: PreTrainedConfig) -> PreTrainedConfig:
        # GB10: the target's own MTP head gets the target's dict overrides (e.g. YaRN rope parameters), then the
        # usual MTP conversion, so its rope table covers max_model_len like the target's.
        def upd(cfg, d):
            for k, v in d.items():
                sub = getattr(cfg, k, None)
                if isinstance(v, dict) and sub is not None and hasattr(sub, "to_dict"):
                    upd(sub, v)
                else:
                    setattr(cfg, k, v)

        upd(hf_config, target_overrides)
        return SpeculativeConfig.hf_config_override(hf_config)

    @staticmethod
    def _apply_composed_hf_override(
"""), (
    """                    draft_hf_overrides = SpeculativeConfig.compose_draft_hf_overrides(
                        self.target_model_config.hf_overrides
                    )
""",
    """                    draft_hf_overrides = SpeculativeConfig.compose_draft_hf_overrides(
                        self.target_model_config.hf_overrides
                    )
                    _gb10_t = self.target_model_config.hf_overrides
                    if self.method == "mtp" and isinstance(_gb10_t, dict) and _gb10_t and self.model == self.target_model_config.model:
                        draft_hf_overrides = functools.partial(SpeculativeConfig._gb10_apply_target_dict, _gb10_t)
                        logger.warning("GB10: MTP draft gets the target's hf_overrides (keys %s)", sorted(_gb10_t))
""")]
for old, new in pairs:
    n = s.count(old)
    if n != 1:
        sys.exit("speculative.py: anchor found %d times: %r" % (n, old[:80]))
    s = s.replace(old, new)
for need in ("import functools", "logger = "):
    if need not in s:
        sys.exit("speculative.py: %r missing" % need)
open(p, "w", encoding="utf-8").write(s)
print("patched config/speculative.py")
