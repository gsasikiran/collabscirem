from __future__ import annotations

from typing import Dict, List, Any

from datasets import load_dataset
from collabscirem.datasets.single_turn import SingleTurnDataset


class SummarizeScientificPapers(SingleTurnDataset):
    """
    Processed OSSAS papers ➟ SingleTurnDataset adaptor (paper Section 4, Figure 4).

    The 700 training papers are rows 0-699 of data/ossas_papers. Each row produced by
    `_preprocess` has at minimum:
        • prompt      - the paper's research question and hypothesis
        • completion  - the paper's methodological details
        • split       - "train" / "test" / "dev" (propagated from HF dataset)
    Plus extra metadata fields (`article_classification`, `type`, …) that SingleTurnDataset
    will automatically place inside the `metadata` column.
    """

    def __init__(
        self,
        repo_id: str = "data/ossas_papers",
        *,
        split: str = "train[:700]",
        eval_ratio: float = 0.2,
        seed: int = 42,
    ):
        raw_ds = load_dataset(repo_id, split=split)  # generated in batches of rows
        processed = self._preprocess(raw_ds)
        super().__init__(processed, eval_ratio=eval_ratio, seed=seed)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _preprocess(raw_ds) -> List[Dict[str, Any]]:
        """
        Convert the original HF splits into the flat dict format expected
        by `SingleTurnDataset`.

        Returns
        -------
        List[dict]
            Items with keys:
                prompt, completion, split, level, type
        """
        processed: List[Dict[str, Any]] = []

        for row in raw_ds:
            # for row in split:
                processed.append(
                    {
                        # required columns
                        "prompt": row["research_question_and_hypothesis"],
                        "completion": row["methodological_details"],
                        # optional / metadata (SingleTurnDataset will group them)
                        "split": "train",
                        "level": row.get("level"),
                        "type": row.get("type"),
                    }
                )

        return processed
