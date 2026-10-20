import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from typing import List, Dict, Any


# --- Strict GPQA CoT Prompting ---
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "gpqa_diamond.txt"), "r") as f:
    GPQA_PROMPT_TEMPLATE = f.read()
# --- Answer Extraction Function (Remains the same) ---
def extract_answer(output: str) -> str | None:
    """
    Extracts the answer letter (A-D) from output formatted like:
    Answer: C   or   Answer: $C
    """
    match = re.search(
        r"(?:Answer\s*[:=]\s*\{?\s*([A-Da-d])\s*\}?)",
        output,
        flags=re.IGNORECASE)
    if match:        
        return match.group(1).upper()
    return None

# --- Step 3: Data Preprocessing (CRITICAL CHANGE HERE) ---
def prepare_gpqa_dataset(dataset_id: str, split: str) -> Dataset:
    """Loads and formats the GPQA dataset, parsing choices from the 'question' column."""
    try:
        dataset = load_dataset(dataset_id, split=split)
    except Exception as e:
        print(f"Error loading GPQA dataset: {e}")
        return Dataset.from_dict({'question':[], 'answer':[]})

    # REGEX PATTERN TO SPLIT THE QUESTION AND OPTIONS
    # This pattern attempts to find the question text followed by the choices starting with 'A)', 'B)', etc.
    # It assumes the options start immediately after the main question text.
    OPTIONS_PATTERN = re.compile(r"([A-D]\s*[\)\.:]\s*.*)", re.DOTALL)
    
    def parse_question_and_options(example):
        full_text = example['question'].strip()
        
        # 1. Find the start of the options
        match = OPTIONS_PATTERN.search(full_text)
        
        if match:
            # The question text is everything before the first option match
            question_text = full_text[:match.start()].strip()
            # The options text is everything from the first option match onwards
            options_text = full_text[match.start():].strip()
        else:
            # If no options found, the whole text is the question, and options are empty
            question_text = full_text
            options_text = ""
            
        return {
            "question_text": question_text,
            "formatted_options": options_text,
            "ground_truth_letter": str(example['answer']).strip().upper() # Use the 'answer' column as the ground truth letter
        }

    dataset = dataset.map(parse_question_and_options)
    
    # Filter out any samples where parsing failed and options are missing (if necessary)
    dataset = dataset.filter(lambda x: len(x["formatted_options"]) > 0)
    
    # Rename columns to match the expected format used in the evaluation loop
    dataset = dataset.rename_column("question", "original_question")
    
    return dataset.select_columns(['question_text', 'formatted_options', 'ground_truth_letter'])


# --- Step 4: Main Evaluation Logic (Updated to use new column names) ---
def run_gpqa_evaluation(model_id: str, dataset_id: str, num_samples: int, data_split: str = "test", seed: int = 42):
    set_seed(seed)
    # Initialize Model and Tokenizer (same as before)
    try:
        # Ensure absolute path so HF doesn't parse this as a repo id
        # import os
        # model_id = os.path.abspath(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}\nTried to load from local path: {model_id}. Ensure the directory contains model weights and tokenizer files.")
        return 0,0,0,0

    # Load and prepare the GPQA dataset
    dataset = prepare_gpqa_dataset(dataset_id, data_split)
    if len(dataset) == 0:
        print("Dataset is empty after loading/filtering.")
        return 0, 0, 0, 0

    dataset_subset = dataset.select(range(min(num_samples, len(dataset))))

    correct_count = 0
    total_count = 0
    token_count = 0
    
    print(f"\n🚀 Starting Evaluation of {model_id} on {len(dataset_subset)} GPQA Diamond problems...")

    for example in tqdm(dataset_subset):
        # USE THE NEWLY PARSED COLUMNS
        if not isinstance(example, dict):
            print("Unexpected row type:", type(example))
            print("Row content:", example)
            raise TypeError("Dataset row is not a dictionary as expected.")

        question = example['question_text']
        ground_truth = example['ground_truth_letter'] 
        options = example['formatted_options']

        # Construct the full prompt
        full_prompt = GPQA_PROMPT_TEMPLATE + f"\n\nQuestion:\n{question}\n\nOptions:\n{options}\n\nProvide a detailed, step-by-step reasoning leading to the answer"

        # Prepare Llama Instruct conversation format
        messages = [
            {"role": "system", "content": "You are an expert scientist with a PhD in Physics, Chemistry, and Biology. Answer the question using complex, multi-step reasoning."},
            {"role": "user", "content": full_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate the response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=4096,
                temperature=0.01,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
        # Count tokens generated
        token_count += output_ids.shape[-1] - input_ids.shape[-1]
        
        # Extract the predicted answer letter
        predicted_answer = extract_answer(output_text)
        print(f"\nQuestion: {question}\nOptions:\n{options}\nPredicted Answer: {predicted_answer}, Ground Truth: {ground_truth}")
        print(f"Full Model Output:\n{output_text}\n{'-'*40}")
        print("-"*80)

        # Scoring: Exact match on the letter
        if predicted_answer == ground_truth:
            correct_count += 1
            # ...
        
        total_count += 1
        
    # --- Step 5: Calculate and Print Final Metric (same as before) ---
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    random_baseline = 100 / 4 # 4 options in MCQ

    average_token_count = round(token_count / total_count, 2) if total_count > 0 else 0
    
    print("\n" + "="*50)
    print(f"📊 Final GPQA Diamond Evaluation Results for {model_id}")
    print(f"Total Problems: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")
    print(f"Random Guessing Baseline: {random_baseline:.2f}%")
    print(f"Average Tokens per Problem: {average_token_count}")
    print("="*50)

    return accuracy, total_count, correct_count, average_token_count
