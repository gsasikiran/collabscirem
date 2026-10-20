# Metric interface from CollabLLM (https://github.com/Wuyxin/collabllm), MIT License.
import logging
from typing import Any, Dict, List, Optional

import openai

from collabscirem.metric import SingleTurnOrChatMetric, BaseMetric
from collabscirem.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
FORMAT_PROMPT = '''You are a helpful and meticulous research methodology evaluator. \
Your task is to evaluate the *format* of the responses provided by an AI assistant \
to the user questions in a given conversation:

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>

You should assess the assistant's response on format to the user's needs. \
Give a float number between 0 and 1, where:
    1 = Properly formatted: The assistant's responses are properly formatted and follow a consistent structure.
     - Example: The assistant provides a well-structured response in the format requested by the user with clear headings, bullet points, or numbered lists.
    0.5 = Moderately formatted: The assistant's responses are somewhat formatted but may lack consistency or clarity.
     - Example: The assistant provides a response that is partially structured but exists inconsistencies in the formatting.
    0 = Poorly formatted: The assistant's responses show little to no proper formatting, making them difficult to read or understand.
     - Example: The assistant provides a response that is poorly structured and lacks any clear formatting.

Output format (JSON):
{{
    "thought": "<How well is the assistant formatted?>",
    "format": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
'''


# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
@SingleTurnOrChatMetric.register_metric("format")
class FormatMetric(BaseMetric):
    """
    Uses an LLM judge to produce a format score in [0, 1].
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 1.0,
            "model": "gpt-5-mini",
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
            raise ValueError("`messages` must be provided for FormatMetric.")

        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )

        eval_prompt = FORMAT_PROMPT.format(chat_history=chat_history)

        logger.debug("Accuracy evaluator prompt:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                response = openai.chat.completions.create(
                    **self.llm_kwargs,
                    messages=[{"role": "user", "content": eval_prompt}]
                )
                full_response = response.choices[0].message.content
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
                if {'thought', 'format'}.issubset(keys):
                    format_score = full_response.pop('format')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return format_score
