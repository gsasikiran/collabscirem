# Metric interface from CollabLLM (https://github.com/Wuyxin/collabllm), MIT License.
import logging
from typing import Any, Dict, List, Optional

import litellm

from collabscirem.metric import SingleTurnOrChatMetric, BaseMetric
from collabscirem.utils.extract_json_reliable import extract_json
from collabscirem.metrics.interactivity import INTERACTIVITY_PROMPT

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
JUSTIFICATION_PROMPT = '''You are a helpful and meticulous conversation evaluator. \
Your task is to evaluate the *justification* of the responses provided by an AI assistant \
to user questions in a given conversation:

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|> 

You should assess the assistant's engagement, clarity, and ability to understand the user's needs. \
Give a float number between 0 and 1, where:
    1 = Highly justified: The assistant provides valid justification for the proposed solution, explaining the reasoning behind its suggestions and proper citations.
        - Example: The assistant thoroughly justifies its recommendations with clear reasoning and references, such as "Based on recent studies (Smith et al., 2020), implementing renewable energy sources can significantly reduce carbon emissions."
    0.5 = Moderately justified: The assistant provides some justification for the proposed solution but may lack evidence or hallucinate the citations.
        - Example: The assistant provides partial justification for its recommendations but lacks strong evidence or proper citations, such as "Renewable energy is beneficial," without citing specific studies.
    0 = Low justification: The assistant provides little to no justification for the proposed solution, failing to explain its reasoning or provide any citations.
        - Example: The assistant makes vague or unsupported claims without any justification, such as "Renewable energy is good," without further explanation or references.


Output format (JSON):
{{
    "thought": "<How justified is the assistant?>",
    "justification": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
'''


# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
@SingleTurnOrChatMetric.register_metric("justification")
class JustificationMetric(BaseMetric):
    """
    Uses an LLM judge to produce a justification score in [0, 1].
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": "claude-3-5-sonnet-latest",
            **llm_kwargs,
        }

    # --------------------------------------------------------------------- #
    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: Optional[str],
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        `prompt`, `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        if not messages:
            raise ValueError("`messages` must be provided for JustificationMetric.")

        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )

        eval_prompt = JUSTIFICATION_PROMPT.format(chat_history=chat_history)

        logger.debug("Accuracy evaluator prompt:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'justification'}.issubset(keys):
                    justification = full_response.pop('justification')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return justification
