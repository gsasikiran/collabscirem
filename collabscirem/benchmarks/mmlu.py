import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset, Dataset
from tqdm import tqdm
from typing import List, Dict, Any

# --- Strict GPQA CoT Prompting ---
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "mmlu.txt"), "r") as f:
    MMLU_PROMPT_TEMPLATE = f.read()
# --- Answer Extraction Function (Remains the same) ---
def extract_answer(output: str) -> str | None:
    """
    Extracts the answer value or letter from output formatted like:
    Answer: 8
    Answer: True, True
    Answer: **C**
    """
    # 1. Look for "Answer" (case insensitive)
    # 2. Skip optional colons, dashes, and spaces
    # 3. Ignore markdown symbols like **, $, or quotes
    # 4. Capture everything until the end of the line
    answer = output.split("Answer: ")[-1].strip()
    if answer:
        # Remove markdown symbols if present
        answer = re.sub(r"^[\*\$\`\"\']+|[\*\$\`\"\']+$", "", answer)
        # Extract only the first line in case of multi-line answers
        answer = answer.splitlines()[0].strip()
        return answer
    return None


# --- Step 3: Data Preprocessing (CRITICAL CHANGE HERE) ---
def prepare_mmlu_dataset(dataset_id: str, split: str) -> Dataset:
    """Loads and formats the MMLU dataset, parsing choices from the 'question' column."""
    try:
        dataset = load_dataset(dataset_id, "all", split="validation")
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        return Dataset.from_dict({'question':[], 'answer':[]})

    # REGEX PATTERN TO SPLIT THE QUESTION AND OPTIONS
    # This pattern attempts to find the question text followed by the choices starting with 'A)', 'B)', etc.
    # It assumes the options start immediately after the main question text.

    
    def parse_question_and_options(example):
        question_text = example['question'].strip()
        
        # 1. Find the start of the options
        options_text = example['choices']
            
        return {
            "question_text": question_text,
            "formatted_options": options_text,
            "ground_truth_letter": str(example['choices'][example['answer']]) # Use the 'answer' column as the ground truth letter
        }

    dataset = dataset.map(parse_question_and_options)
    
    # Filter out any samples where parsing failed and options are missing (if necessary)
    dataset = dataset.filter(lambda x: len(x["formatted_options"]) > 0)
    
    # Rename columns to match the expected format used in the evaluation loop
    # dataset = dataset.rename_column("question", "original_question") 
    
    return dataset.select_columns(['question_text', 'formatted_options', 'ground_truth_letter'])


# --- Step 4: Main Evaluation Logic (Updated to use new column names) ---
def run_mmlu_evaluation(model_id: str, dataset_id: str, data_split: str = "validation", seed: int = 42):
    set_seed(seed)
    # Initialize Model and Tokenizer (same as before)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
            device_map="auto"          
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0,0,0,0

    # Load and prepare the MMLU dataset
    dataset = prepare_mmlu_dataset(dataset_id, data_split)
    if len(dataset) == 0:
        print("Dataset is empty after loading/filtering.")
        return 0, 0, 0, 0

    dataset_subset = dataset

    correct_count = 0
    total_count = 0

    token_count = 0
    
    print(f"\n🚀 Starting Evaluation of {model_id} on {len(dataset_subset)} MMLU problems...")

    for example in tqdm(dataset_subset):
        # USE THE NEWLY PARSED COLUMNS
        question = example['question_text']
        ground_truth = example['ground_truth_letter'].lower().strip()
        options = example['formatted_options']


        # Construct the full prompt
        full_prompt = MMLU_PROMPT_TEMPLATE + f"\n\nQuestion:\n{question}\n\nOptions:\n{options}\n\nProvide a detailed, step-by-step reasoning leading to the answer"

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
        predicted_answer = predicted_answer.lower().strip() if predicted_answer else ""
        print(f"\nQuestion: {question}\nOptions:\n{options}\nPredicted Answer: {predicted_answer}, Ground Truth: {ground_truth}")
        print(f"Full Model Output:\n{output_text}\n{'-'*40}")
        print("-"*80)

        # Scoring: Exact match on the letter
        if predicted_answer == ground_truth:
            correct_count += 1
            print("✅ Correct Answer!")
        else:
            print("❌ Incorrect Answer.")
            # ...
        
        total_count += 1
        
    # --- Step 5: Calculate and Print Final Metric (same as before) ---
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    random_baseline = 100 / 4 # 4 options in MCQ

    average_token_count = round(token_count / total_count, 2) if total_count > 0 else 0
    
    print("\n" + "="*50)
    print(f"📊 Final MMLU Evaluation Results for {model_id}")
    print(f"Total Problems: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")
    print(f"Average Tokens per Problem: {average_token_count}")
    print(f"Random Guessing Baseline: {random_baseline:.2f}%")
    print("="*50)

    return accuracy, total_count, correct_count, average_token_count