"""BM25 with pseudo-relevance feedback query expansion and RRF fusion.

Mechanism (distinct from baselines):
- First BM25 pass on the raw question.
- Take top anchor turns and mine their high-IDF tokens (pseudo-relevance
  feedback). This is a classical IR technique that is not present in the
  existing BM25, Mem0, or A-Mem scaffolds.
- Second BM25 pass with the expanded query.
- Fuse both rankings with Reciprocal Rank Fusion; return the top turns
  without window expansion so token budget stays tight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memomemo.schemas import ConversationTurn, LocomoExample, RetrievalHit
from memomemo.scaffolds.base import RetrievalMemoryScaffold, ScaffoldConfig
from memomemo.scaffolds.bm25_scaffold import SimpleBM25Okapi
from memomemo.utils.text import tokenize


_MAX_DOC_FRAC = 0.3
_DEFAULT_PRF_K = 3
_DEFAULT_PRF_EXPAND = 8
_RRF_K = 60
_RRF_WEIGHT_PRIMARY = 1.0
_RRF_WEIGHT_EXPANDED = 0.7


@dataclass
class RrfPrfState:
    turns: tuple[ConversationTurn, ...]
    docs: list[str]
    tokenized_docs: list[list[str]]
    bm25: Any


class RrfPrfBm25Scaffold(RetrievalMemoryScaffold):
    """Two-pass BM25 with pseudo-relevance feedback + reciprocal rank fusion."""

    name = "rrf_prf_bm25"
    reference_urls = (
        "https://github.com/dorianbrown/rank_bm25.git",
    )

    def build(self, example: LocomoExample, config: ScaffoldConfig) -> RrfPrfState:
        docs = [turn.render() for turn in example.conversation]
        tokenized_docs = [tokenize(doc) for doc in docs]
        bm25 = SimpleBM25Okapi(tokenized_docs)
        return RrfPrfState(
            turns=example.conversation,
            docs=docs,
            tokenized_docs=tokenized_docs,
            bm25=bm25,
        )

    def retrieve(
        self,
        state: RrfPrfState,
        question: str,
        config: ScaffoldConfig,
    ) -> list[RetrievalHit]:
        query = tokenize(question)
        if not query or not state.turns:
            return []

        prf_k = int(config.extra.get("prf_k", _DEFAULT_PRF_K))
        prf_expand = int(config.extra.get("prf_expand", _DEFAULT_PRF_EXPAND))
        max_doc_frac = float(config.extra.get("max_doc_frac", _MAX_DOC_FRAC))

        primary_scores = list(state.bm25.get_scores(query))
        primary_rank = sorted(
            range(len(state.turns)),
            key=lambda idx: primary_scores[idx],
            reverse=True,
        )

        expansion = self._mine_expansion(
            state=state,
            query=query,
            primary_rank=primary_rank,
            primary_scores=primary_scores,
            prf_k=prf_k,
            prf_expand=prf_expand,
            max_doc_frac=max_doc_frac,
        )

        if expansion:
            expanded_query = query + expansion
            expanded_scores = list(state.bm25.get_scores(expanded_query))
        else:
            expanded_scores = primary_scores
        expanded_rank = sorted(
            range(len(state.turns)),
            key=lambda idx: expanded_scores[idx],
            reverse=True,
        )

        fused: dict[int, float] = {}
        for rank_pos, idx in enumerate(primary_rank):
            fused[idx] = fused.get(idx, 0.0) + _RRF_WEIGHT_PRIMARY / (_RRF_K + rank_pos)
        for rank_pos, idx in enumerate(expanded_rank):
            fused[idx] = fused.get(idx, 0.0) + _RRF_WEIGHT_EXPANDED / (_RRF_K + rank_pos)

        ranked = sorted(fused, key=lambda idx: fused[idx], reverse=True)
        picks: list[int] = []
        for idx in ranked:
            if primary_scores[idx] <= 0 and expanded_scores[idx] <= 0:
                continue
            picks.append(idx)
            if len(picks) >= max(1, config.top_k):
                break

        picks.sort()  # preserve chronological order for the answering prompt
        return [
            RetrievalHit(
                text=state.docs[idx],
                score=float(fused[idx]),
                source=self.name,
                metadata={
                    "turn_index": idx,
                    "primary_score": float(primary_scores[idx]),
                    "expanded_score": float(expanded_scores[idx]),
                    "expansion": list(expansion),
                },
            )
            for idx in picks
        ]

    def _mine_expansion(
        self,
        *,
        state: RrfPrfState,
        query: list[str],
        primary_rank: list[int],
        primary_scores: list[float],
        prf_k: int,
        prf_expand: int,
        max_doc_frac: float,
    ) -> list[str]:
        n_docs = max(1, len(state.turns))
        df_cap = max(1, int(n_docs * max_doc_frac))
        query_set = set(query)
        token_weight: dict[str, float] = {}
        seen_anchors = 0
        for idx in primary_rank:
            if seen_anchors >= prf_k:
                break
            if primary_scores[idx] <= 0:
                break
            seen_anchors += 1
            for token in set(state.tokenized_docs[idx]):
                if token in query_set:
                    continue
                df = state.bm25.doc_freq.get(token, 0)
                if df == 0 or df > df_cap:
                    continue
                token_weight[token] = token_weight.get(token, 0.0) + 1.0 / (df + 1)
        return sorted(token_weight, key=lambda tok: token_weight[tok], reverse=True)[:prf_expand]
