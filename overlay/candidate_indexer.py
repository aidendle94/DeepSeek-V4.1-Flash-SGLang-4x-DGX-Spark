"""Keep DeepSeek-V4.1's candidate-block indexer on the dense path on SM12x (GB10, RTX PRO 6000 Blackwell).

SGLang builds the candidate indexer (the layer that publishes the top candidate blocks the later sparse layers read)
with `v41_indexer.make_candidate_indexer`. On any device of compute capability 10 or newer it now picks the sparse-table
backend, which needs DeepGEMM's paged sparse MQA logits; SGLang enables those only on SM100
(`DEEPGEMM_PAGED_SPARSE_MQA_LOGITS` requires `DEEPGEMM_BLACKWELL`), so on SM12x the factory raises at backend init
("the candidate indexer needs DeepGEMM's paged sparse MQA logits"). The dense-blocks backend the same factory returns
below SM100 is what SM12x ran before that change: DeepGEMM's dense fp8/fp4 logits for prefill (unless
SGLANG_DSV41_TORCH_PREFILL_INDEXER is set) and torch block selection for decode.

This hook wraps the backend module's binding of the factory: on SM12x it returns the dense-blocks backend for both
prefill and decode, built with the factory's own arguments and its own DeepGEMM-prefill decision; on every other device
it calls the stock factory. The capability is read when the backend is built (inside the worker, CUDA already set up),
never at import.

  SPARK_CANDIDATE_DENSE   1 (default): dense-blocks candidate indexer on SM12x; 0: SGLang's choice

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def install(module) -> None:
    """sglang.srt.layers.attention.deepseek_v4_backend: wrap its make_candidate_indexer."""
    if os.environ.get("SPARK_CANDIDATE_DENSE", "1") != "1":
        return
    stock = module.make_candidate_indexer
    from sglang.srt.layers.attention.dsv4 import v41_indexer
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import DenseBlocksBackend

    def make_candidate_indexer(*, token_to_kv_pool, req_to_token, page_size, candidate_topk_blocks,
                               candidate_block_size, **extra):
        if extra or torch.cuda.get_device_capability()[0] != 12:
            return stock(token_to_kv_pool=token_to_kv_pool, req_to_token=req_to_token, page_size=page_size,
                         candidate_topk_blocks=candidate_topk_blocks, candidate_block_size=candidate_block_size, **extra)
        use_deep_gemm_prefill = v41_indexer._use_deep_gemm_prefill()
        dense = DenseBlocksBackend(
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            use_deep_gemm_prefill=use_deep_gemm_prefill,
        )
        logger.info("candidate indexer on SM12x: dense blocks for prefill and decode (DeepGEMM prefill logits: %s)",
                    use_deep_gemm_prefill)
        return dense, dense

    module.make_candidate_indexer = make_candidate_indexer
