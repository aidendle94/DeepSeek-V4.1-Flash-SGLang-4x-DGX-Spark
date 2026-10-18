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

Decode scoring. Below SM100 the dense-blocks backend scores a decode row with `fp4_index_logits_decode`, a Triton
kernel that reads every visible compressed slot through the page table with bf16 queries; on GB10 at a 64K context the
indexers of one decode step took 10.2 ms with it, against 0.9 ms on the previous base, which scored the same rows with
DeepGEMM's paged fp4 MQA logits (`fp8_fp4_paged_mqa_logits`, fp4 queries; the DeepGEMM plan is forced for compress
ratios 1 and 2 on SM120 by indexer_schedule). With SPARK_CANDIDATE_DG_DECODE=1 the backend's decode scores come from
that DeepGEMM call, made exactly as SGLang's full top-k indexer makes it, sliced to the same width, with positions at or
past each row's length set to -inf as the Triton kernel returns them; slots, block selection and top-k are SGLang's.
The scores are the previous base's (fp4 queries), not bit-identical to the Triton kernel's (bf16 queries). Rows the
metadata splits into chunks (a list of DeepGEMM plans) and SM90 keep the stock scorer.

  SPARK_CANDIDATE_DENSE      1 (default): dense-blocks candidate indexer on SM12x; 0: SGLang's choice
  SPARK_CANDIDATE_DG_DECODE  1 (default): DeepGEMM paged logits for its decode scores on SM12x; 0: SGLang's Triton kernel

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
    if os.environ.get("SPARK_CANDIDATE_DG_DECODE", "1") == "1":
        _install_dg_decode()


def _install_dg_decode() -> None:
    """Wrap dense_blocks' binding of scoring.decode_scores (see the module docstring)."""
    from sglang.kernels.ops.attention.dsv4.index_logits import deep_gemm_fp4_paged_mqa_logits
    from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DecodeScores, get_deep_gemm_decode_data

    stock = dense_blocks.decode_scores
    state = {"logged": False}

    def decode_scores(*, inputs, token_to_kv_pool, req_to_token, candidate_mask=None, compute_topk_plan=True, **extra):
        md = inputs.paged_metadata
        dgm = getattr(md, "deep_gemm_metadata", None)
        if (extra or dgm is None or isinstance(dgm, list) or torch.cuda.get_device_capability()[0] != 12):
            return stock(inputs=inputs, token_to_kv_pool=token_to_kv_pool, req_to_token=req_to_token,
                         candidate_mask=candidate_mask, compute_topk_plan=compute_topk_plan, **extra)
        req, pos = inputs.req_rows, inputs.positions
        bs = req.shape[0]
        assert pos.shape[0] == bs, f"decode expects one token per request, {pos.shape=} {bs=}"
        if bs == 0:
            inputs.reset_outputs()
            return None
        ratio = inputs.compress_ratio
        lens = (pos + 1) // ratio
        lmax = min(md.max_compressed_seq_len, req_to_token.shape[1] // ratio)
        if lmax == 0:
            inputs.reset_outputs()
            return None
        data = get_deep_gemm_decode_data(inputs, token_to_kv_pool)
        logits = deep_gemm_fp4_paged_mqa_logits((data.q_fp4, data.q_sf), data.k_cache, data.weights,
                                                md.compressed_seq_lens, md.page_table, dgm, md.max_compressed_seq_len)
        inputs.reset_outputs()
        j = torch.arange(lmax, device=pos.device)
        valid = j[None, :] < lens[:, None]
        slots = req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
        slots = slots.masked_fill(~valid, 0)
        scores = logits[:, :lmax].float().masked_fill(~valid, float("-inf")).contiguous()
        if not state["logged"]:
            state["logged"] = True
            logger.info("candidate indexer decode scores on SM12x: DeepGEMM paged fp4 logits (ratio %d, width %d)", ratio, lmax)
        return DecodeScores(bs=bs, lmax=lmax, lens=lens, slots=slots, scores=scores)

    dense_blocks.decode_scores = decode_scores
