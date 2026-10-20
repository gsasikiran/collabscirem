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
CLARITY_PROMPT = '''You are a helpful and meticulous research methodology evaluator. \
Your task is to evaluate the *clarity* of the responses provided by an AI assistant \
to the user questions in a given conversation:

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>

You should assess the assistant's response on clarity to the user's needs. \
Give a float number between 0 and 1, where:
    1 = Highly clear: The assistant's responses are very clear and easy to understand, providing relevant and accurate information that directly addresses the user's questions.
     - Example: The assistant thoroughly understands the user's question and provides a detailed, accurate response that directly addresses the user's intent, such as "Based on your question about climate change, here are the key factors contributing to it..."
    0.5 = Moderately clear: The assistant's responses are somewhat clear but may miss some key aspects or provide partially relevant information.
     - Example: The assistant provides a response that addresses part of the user's question but misses some important details, such as "Climate change is influenced by various factors," without elaborating on the specifics.
    0 = Low clarity: The assistant's responses show little to no clarity with the user's needs, providing irrelevant or inaccurate information that does not address the user's questions.
     - Example: The assistant provides a vague or off-topic response that does not relate to the user's question, such as "Climate change is bad," without any further context or relevance to the user's inquiry.

Output format (JSON):
{{
    "thought": "<How clear is the assistant?>",
    "clarity": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
'''


# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
@SingleTurnOrChatMetric.register_metric("clarity")
class ClarityMetric(BaseMetric):
    """
    Uses an LLM judge to produce a clarity score in [0, 1].
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
            raise ValueError("`messages` must be provided for ClarityMetric.")

        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )

        eval_prompt = CLARITY_PROMPT.format(chat_history=chat_history)

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
                if {'thought', 'clarity'}.issubset(keys):
                    clarity = full_response.pop('clarity')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return clarity
