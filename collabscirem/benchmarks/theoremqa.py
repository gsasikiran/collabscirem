import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, set_seed
from tqdm import tqdm

def run_theorem_qa_evaluation(model_id, seed: int = 42):
    set_seed(seed)

    # 1. Load the dataset
    print("Loading TheoremQA dataset...")
    dataset = load_dataset("TIGER-Lab/TheoremQA", split="test")

    # 2. Filter out rows that have images
    # Text-only questions have None or an empty string in the 'picture' column
    dataset = dataset.filter(lambda x: x['Picture'] is None or x['Picture'] == "")
    print(f"Filtered dataset: {len(dataset)} text-only questions remaining.")

    # 3. Initialize Tokenizer and Pipeline
    print(f"Loading model and tokenizer {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True},
        device_map="auto",
    )

    correct = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_samples = len(dataset)

    print(f"Starting evaluation...")

    for i, item in enumerate(tqdm(dataset)):
        question = item['Question']
        # TheoremQA answers are usually specific numerical values or short strings
        ground_truth = str(item['Answer']).strip().lower()

        prompt_content = (
            f"Question: {question}\n"
            # f"Theorem/Field: {item['Theorem']} ({item['Field']})\n"
            f"Task: Solve the problem above. Provide the final numerical or short-text answer clearly. "
            f"Finish your response with 'Final Answer: <answer>'."
        )

        messages = [
            {"role": "system", "content": "You are a mathematical expert. Solve the theorem-based problem step by step and provide the final result."},
            {"role": "user", "content": prompt_content},
        ]

        # Token Counting: Input
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        total_input_tokens += input_ids.shape[1]

        # Inference
        outputs = pipe(
            messages,
            max_new_tokens=4096, # TheoremQA may need Chain-of-Thought space
            do_sample=True,
            temperature=0.01,   # Low temperature for deterministic math
            pad_token_id=tokenizer.eos_token_id
        )

        # Token Counting: Output
        response_text = outputs[0]["generated_text"][-1]["content"]
        output_ids = tokenizer.encode(response_text, add_special_tokens=False)
        total_output_tokens += len(output_ids)

        # Accuracy Check (Simple exact match on the final part of the string)
        # Note: Math evaluation often requires more complex parsing for floats/fractions
        prediction = response_text.lower()
        if "final answer:" in prediction:
            prediction = prediction.split("final answer:")[-1].strip()

        is_correct = ground_truth in prediction
        if is_correct:
            correct += 1

        print(f"\nQuestion: {question}\nPredicted Answer: {prediction}, Ground Truth: {ground_truth}, Correct: {is_correct}")
        print(f"Full Model Output:\n{response_text}\n{'-'*40}")
        print("-"*80)

    # Final Statistics
    final_accuracy = (correct / total_samples) * 100
    print(f"\n--- TheoremQA Results (Text Only) ---")
    print(f"Total Questions:      {total_samples}")
    print(f"Correct (Est.):       {correct}")
    print(f"Accuracy:             {final_accuracy:.2f}%")
    print(f"--- Token Statistics ---")
    print(f"Total Input Tokens:   {total_input_tokens}")
    print(f"Total Output Tokens:  {total_output_tokens}")
    print(f"Avg Tokens/Q:         {(total_output_tokens) / total_samples:.1f}")
    return final_accuracy, total_samples, correct, (total_output_tokens) / total_samples