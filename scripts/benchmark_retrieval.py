"""Compare repeated BM25 preparation with cached statistics; no providers required.

Run with ``uv run python scripts/benchmark_retrieval.py``. This measures only
sparse ranking over synthetic text, not database, dense retrieval or model time.
"""

import json
from datetime import date
from statistics import median
from time import perf_counter

from fra.domain import EvidenceChunk
from fra.retrieval.hybrid import BM25Okapi, _sparse_index, _sparse_ranking, _tokenize


def uncached_ranking(chunks, query):
    scores = BM25Okapi([_tokenize(chunk.content) for chunk in chunks]).get_scores(query)
    return [
        chunk.id
        for chunk, _ in sorted(
            zip(chunks, scores, strict=True), key=lambda item: (-item[1], item[0].id)
        )
    ][:100]


def main() -> None:
    chunks = []
    for index in range(1000):
        content = (f"Revenue cash operational risk {index} " * 60).strip()
        chunks.append(
            EvidenceChunk(
                id=f"c-{index}",
                ticker="NVDA",
                corpus_version="bench-v1",
                content=content,
                source_url="https://www.sec.gov/test",
                form="10-Q",
                filed_at=date(2026, 1, 1),
                accession_no="test",
                section="MD&A",
                raw_start=0,
                raw_end=len(content),
            )
        )
    queries = [[term] for term in ("revenue", "cash", "risk", "operational", "500")] * 4
    for query in queries:
        assert uncached_ranking(chunks, query) == _sparse_ranking(chunks, query)

    def timed(rank):
        started = perf_counter()
        for query in queries:
            rank(chunks, query)
        return perf_counter() - started

    baseline = median(timed(uncached_ranking) for _ in range(3))
    _sparse_index.cache_clear()
    started = perf_counter()
    _sparse_ranking(chunks, queries[0])
    cold = perf_counter() - started
    warm = median(timed(_sparse_ranking) for _ in range(3))
    print(
        json.dumps(
            {
                "chunks": len(chunks),
                "queries_per_trial": len(queries),
                "trials": 3,
                "baseline_median_s": baseline,
                "warm_median_s": warm,
                "cold_first_query_s": cold,
                "warm_speedup": baseline / warm,
                "identical_rankings": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
