# Adapted from CollabLLM (https://github.com/Wuyxin/collabllm), MIT License;
# extended here for the science-aware reward decomposition.
"""
collabscirem.datasets.multiturn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified loader + wrapper for multi-turn chat data.

Initialization supports three input styles:

1. **Flat list** (`List[dict]`) with required keys per row:
   {'prompt', 'completion', 'conv_id', 'score',
    'single_turn_prompt', 'single_turn_completion', 'single_turn_metadata'}

2. **Nested structure** (`List[dict]`) per conversation:
   [
     {
       "conv_id": ...,
       "single_turn_prompt": ...,
       "single_turn_completion": ...,
       "single_turn_metadata": ...,
       "turns": [
         {
           "prompt": [ {role,content}, ... ],
           "responses": [
             {"completion": ..., "score": ...},
             {"completion": ..., "score": ...},
             ...
           ]
         },
         ...
       ]
     },
     ...
   ]

3. **Local HF dataset directory** (path) or **HF Hub repo ID** (string).

In all cases, the internal `self.data` will be a flat `List[dict]` with keys:
{prompt, completion, conv_id, score, single_turn_prompt, single_turn_completion,
 single_turn_metadata, turn_id}

Derived field
-------------
• `turn_id` is set to `len(prompt)` if not provided explicitly.

Converters (all use a uniform random splitter)
-----------------------------------------------
• `to_sft_dataset()`   → DatasetDict {text}  
• `to_dpo_dataset()`   → DatasetDict {prompt, chosen, rejected, score_*}  
• `to_inputs_dataset()`→ DatasetDict {prompt, single_turn_*}
"""

from __future__ import annotations

import os
import random
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from collabscirem.prompts import SYSTEM_PROMPT
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

_REQUIRED: set[str] = {
    "prompt",
    "completion",
    "conv_id",
    "score",
    "single_turn_prompt",
    "single_turn_completion",
    "single_turn_metadata",
}

import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# uniform splitter                                                            #
# --------------------------------------------------------------------------- #
def _uniform_split(
    full_ds: Dataset,
    *,
    eval_ratio: float,
    n_eval: Optional[int],
    seed: int,
) -> DatasetDict:
    k = n_eval if n_eval is not None else int(eval_ratio * len(full_ds))
    k = min(k, len(full_ds))

    random.seed(seed)
    eval_idx = set(random.sample(range(len(full_ds)), k=k))
    train_idx = [i for i in range(len(full_ds)) if i not in eval_idx]

    return DatasetDict(
        {
            "train": full_ds.select(train_idx),
            "eval": full_ds.select(sorted(eval_idx)),
        }
    )


# --------------------------------------------------------------------------- #
# main dataclass                                                              #
# --------------------------------------------------------------------------- #
class MultiturnDataset:
    def __init__(
        self,
        data_or_local_dir_or_hf_repo_or_nested: Union[List[Dict[str, Any]], str],
        *,
        seed: int = 42,
        add_system_prompt: bool = True,
    ):
        """
        Parameters
        ----------
        data_or_local_dir_or_hf_repo_or_nested :
            • Flat list of dicts with required keys (old style), OR
            • Nested list of conversations (new style), OR
            • Local path saved by `Dataset.save_to_disk`, OR
            • HF Hub repo ID (e.g. "org/dataset").
        seed : int
            RNG seed for uniform splitting.
        """
        self.seed = seed
        self.sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}] if add_system_prompt else []

        # 1) Load raw data into `raw_list` of dicts
        if isinstance(data_or_local_dir_or_hf_repo_or_nested, list):
            raw_list = data_or_local_dir_or_hf_repo_or_nested
        elif any(os.path.exists(os.path.join(str(data_or_local_dir_or_hf_repo_or_nested), f))
                 for f in ("dataset_dict.json", "state.json")):
            ds_dict = load_from_disk(str(data_or_local_dir_or_hf_repo_or_nested))  # type: ignore
            raw_list = [dict(r) for r in ds_dict.flatten()]
        else:
            ds_dict = load_dataset(str(data_or_local_dir_or_hf_repo_or_nested), trust_remote_code=True)  # type: ignore
            raw_list = [dict(r) for _, split in ds_dict.items() for r in split]

        if not raw_list:
            raise ValueError("Loaded dataset is empty.")

        # 2) Detect nested structure: presence of "turns" key in first element
        if isinstance(raw_list[0], dict) and "turns" in raw_list[0]:
            self.data = self._flatten_nested(raw_list)
        else:
            # Assume flat structure; validate required keys
            if not _REQUIRED.issubset(raw_list[0]):
                missing = _REQUIRED - set(raw_list[0])
                raise ValueError(f"Missing required keys in flat data: {missing}")

            # Auto-fill turn_id if missing
            for row in raw_list:
                if not isinstance(row["prompt"], Sequence):
                    raise TypeError("`prompt` must be a list of messages.")
                row.setdefault("turn_id", len(row["prompt"]))

            self.data = raw_list  # type: ignore

        if not self.data:
            raise ValueError("No valid rows after processing input.")

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        token: Optional[str] = None,
        split: Optional[str] = None,
    ) -> DatasetDict:

        """
        Push the dataset to the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The repository ID on the Hugging Face Hub.
        private : bool
            Whether to create a private repository.
        token : Optional[str]
            Optional authentication token for the Hub.
        split : Optional[str]
            If provided, will save only this split (e.g., "train", "eval").

        Returns
        -------
        DatasetDict
            The pushed dataset.
        """
        ds = Dataset.from_dict({k: [row[k] for row in self.data] for k in self.data[0]})
        return ds.push_to_hub(repo_id, private=private, token=token, split=split)


    def _flatten_nested(self, nested: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert nested conversation format to flat list of rows.

        Nested format per conversation:
        {
          "conv_id": ...,
          "single_turn_prompt": ...,
          "single_turn_completion": ...,
          "single_turn_metadata": ...,
          "turns": [
             {
               "prompt": [...],
               "responses": [
                  {"completion": ..., "score": ..., **kwargs}, ...
               ]
             },
             ...
          ]
        }

        Output per row:
        {
          "prompt": [...],
          "completion": ...,
          "conv_id": ...,
          "score": ...,
          "single_turn_prompt": ...,
          "single_turn_completion": ...,
          "single_turn_metadata": ...,
          "turn_id": len(prompt)
        }
        """
        flat = []
        for base_conv_id, convo in enumerate(nested):
            # Validate presence of required conversation-level keys
            for key in {"single_turn_prompt", "single_turn_completion", "single_turn_metadata", "turns"}:
                if key not in convo:
                    raise ValueError(f"Missing key '{key}' in nested conversation.")
            st_prompt = convo["single_turn_prompt"]
            st_completion = convo["single_turn_completion"]
            st_metadata = convo["single_turn_metadata"]

            for turn in convo["turns"]:
                if "prompt" not in turn or "responses" not in turn:
                    raise ValueError("Each turn must have 'prompt' and 'responses'.")
                prompt_msgs = turn["prompt"]
                if not isinstance(prompt_msgs, Sequence):
                    raise TypeError("`turn['prompt']` must be a list of messages.")
                turn_id = len(prompt_msgs)
                for resp in turn["responses"]:
                    if "completion" not in resp or "score" not in resp:
                        raise ValueError("Each response must have 'completion' and 'score'.")
                    flat.append(
                        {
                            "prompt": prompt_msgs,
                            "completion": resp["completion"],
                            "conv_id": base_conv_id,
                            "score": resp["score"],
                            "single_turn_prompt": st_prompt,
                            "single_turn_completion": st_completion,
                            "single_turn_metadata": st_metadata,
                            "turn_id": turn_id,
                            **{k: resp.get(k) for k in resp if k not in {"completion", "score"}},
                        }
                    )
        return flat

    def _get_score_from_row(
        self,
        row: Dict[str, Any],
        *,
        score_key: Optional[str] = None,
        score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> float:
        """Compute a numeric score for a row.

        If ``score_fn`` is provided it is called with the row and its return
        value is used. Otherwise, if ``score_key`` is provided it supports a
        dotted path like ``rewards.MR`` or a top-level key. Falls back to the
        existing ``row['score']`` when a value isn't found or conversion fails.
        """
        if score_fn is not None:
            try:
                print("Using score_fn to compute score")
                result = score_fn(row)
                print("score_fn result:", result)
                return float(result)
            except Exception:
                logger.exception("score_fn raised an error; falling back to row['score']")

        if not score_key:
            return float(row.get("score", 0.0))

        # support dotted path e.g. 'rewards.MR'
        cur: Any = row
        for part in score_key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break

        if cur is None:
            # try shortcuts: direct key or inside rewards dict
            cur = row.get(score_key)
            if cur is None and isinstance(row.get("rewards"), dict):
                cur = row["rewards"].get(score_key)

        try:
            # If the extracted value is array-like, return its mean (matches prior behaviour)
            arr = np.asarray(cur)
            if arr.size > 1:
                print("Using mean of array-like score for key", score_key)
                return float(arr.mean())
            # scalar-like
            return float(arr.item())
        except Exception:
            logger.warning("Failed to parse score for key '%s', using row['score']", score_key)
            # fallback: try to coerce row['score'] (may itself be array-like)
            try:
                s = np.asarray(row.get("score", 0.0))
                if s.size > 1:
                    return float(s.mean())
                return float(s.item())
            except Exception:
                return 0.0

    # ------------------------------------------------------------------ #
    # SFT                                                                #
    # ------------------------------------------------------------------ #
    def to_sft_dataset(
        self,
        *,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
        lower_bound_metric: Optional[str] = None,
        lower_bound: Optional[float] = 0.0,
        score_key: Optional[str] = None,
        score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> DatasetDict:

        # Select best example per conversation ID: prefer latest turn, then highest score
        best_examples: Dict[int, tuple] = {}
        for row in self.data:
            cid = row["conv_id"]
            score_val = float(self._get_score_from_row(row, score_key=score_key, score_fn=score_fn))
            print("Row conv_id:", cid, "turn_id:", row["turn_id"], "score:", score_val)
            prev = best_examples.get(cid)
            if prev is None:
                best_examples[cid] = (row, score_val)
                continue

            prev_row, prev_score = prev
            # prefer later turn; if same turn prefer higher computed score
            if row["turn_id"] > prev_row["turn_id"] or (
                row["turn_id"] == prev_row["turn_id"] and score_val > prev_score
            ):
                best_examples[cid] = (row, score_val)

        # Build SFT dialogues, filtering by optional metric threshold
        serialized_dialogues = []
        for row_entry in best_examples.values():
            row = row_entry[0]
            if lower_bound_metric:
                try:
                    metric = row
                    for key in lower_bound_metric.split("."):
                        metric = metric.get(key, {})
                    value = np.asarray(metric).mean().item()
                except Exception as e:
                    logger.error(f"Failed to extract metric '{lower_bound_metric}' from row: {row} — {e}")
                    continue

                if value < lower_bound:
                    logger.warning(
                        f"Filtered out conv_id={row['conv_id']} (turn_id={row['turn_id']}) "
                        f"due to {lower_bound_metric}={value:.3f} < {lower_bound:.3f}"
                    )
                    continue

            if not isinstance(row["prompt"], list):
                raise TypeError("Expected `prompt` to be a list of messages.")

            messages = self.sys_msg + row["prompt"] + [{"role": "assistant", "content": row["completion"]}]
            serialized_dialogues.append(messages)

        logger.info(
            f"Converted {len(serialized_dialogues)} dialogues "
            f"(filter: {lower_bound_metric} ≥ {lower_bound}); "
            f"retention ratio: {len(serialized_dialogues)/len(best_examples):.2f}"
        )

        full_dataset = Dataset.from_dict({"messages": serialized_dialogues})
        return _uniform_split(full_dataset, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    # ------------------------------------------------------------------ #
    # DPO                                                                #
    # ------------------------------------------------------------------ #
    def to_dpo_dataset(
        self,
        *,
        minimum_gap: float = 0.0,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
        score_key: Optional[str] = None,
        score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> DatasetDict:

        # Helper to compute the score to use (original `score` by default)
        def _row_score(r: Dict[str, Any]) -> float:
            return float(self._get_score_from_row(r, score_key=score_key, score_fn=score_fn))

        # Group rows by (conv_id, turn_id)
        grouped: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in self.data:
            grouped.setdefault((r["conv_id"], r["turn_id"]), []).append(r)

        pairs = []
        for items in grouped.values():
            if len(items) < 2:
                continue
            scored = [(r, _row_score(r)) for r in items]
            scored.sort(key=lambda t: t[1], reverse=True)
            if scored[0][1] - scored[-1][1] < minimum_gap:
                continue
            top_r, top_s = scored[0]
            bot_r, bot_s = scored[-1]
            pairs.append(
                {
                    "prompt": self.sys_msg + top_r["prompt"],
                    "chosen": top_r["completion"],
                    "rejected": bot_r["completion"],
                    "score_chosen": top_s,
                    "score_rejected": bot_s,
                }
            )
            # logger.info(f"Formed DPO pair conv_id: {top_r['conv_id']}, turn_id: {top_r['turn_id']}, "
            #       f"         score gap: {top_s - bot_s}")
        logger.info(f"{pairs[:5]} ...")  # log first few pairs for sanity check
        logger.info(f"Converted {len(pairs)} pairs (minimum_gap={minimum_gap}, ratio={len(pairs)/len(self.data):.2f})")

        if not pairs:
            return DatasetDict({"train": Dataset.from_dict({}), "eval": Dataset.from_dict({})})

        full_ds = Dataset.from_dict({k: [p[k] for p in pairs] for k in pairs[0]})
        return _uniform_split(full_ds, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    # ------------------------------------------------------------------ #
    # Inputs                                                             #
    # ------------------------------------------------------------------ #
    def to_inputs_dataset(
        self,
        *,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
        score_key: Optional[str] = None,
        score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> DatasetDict:

        # Keep exactly one row per (conv_id, turn_id)
        unique: Dict[tuple, Dict[str, Any]] = {}
        for r in self.data:
            key = (r["conv_id"], r["turn_id"])
            if key not in unique:
                unique[key] = r
            r['prompt'] = self.sys_msg + r["prompt"]

        keep_keys = [
            "prompt",
            "single_turn_prompt",
            "single_turn_completion",
            "single_turn_metadata",
            "score",
            "rewards"
        ]
        # If requested, compute a score from rewards (or other fields) and expose
        records = []
        for row in unique.values():
            rec = {k: row[k] for k in keep_keys}
            if score_key is not None or score_fn is not None:
                rec['score'] = float(self._get_score_from_row(row, score_key=score_key, score_fn=score_fn))
            records.append(rec)
        if not records:
            return DatasetDict({"train": Dataset.from_dict({}), "eval": Dataset.from_dict({})})

        full_ds = Dataset.from_dict({k: [rec[k] for rec in records] for k in keep_keys})
        return _uniform_split(full_ds, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    # ------------------------------------------------------------------ #
    # misc                                                               #
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
