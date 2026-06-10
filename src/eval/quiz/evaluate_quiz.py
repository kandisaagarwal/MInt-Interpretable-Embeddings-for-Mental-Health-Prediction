import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

import re
def choose_outlier_batch(options_list):
    prompts = [
        f"""You are taking an outlier detection quiz.

Three items belong to the same group.
One item does not belong.

Reply with ONLY the number (1-4) of the outlier and nothing else.

1. {opts[0]}
2. {opts[1]}
3. {opts[2]}
4. {opts[3]}

Your answer: """ for opts in options_list
    ]
    # Tokenize as a batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode and extract answers
    answers = []
    for i in range(len(prompts)):
        gen_text = tokenizer.decode(generated_ids[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        match = re.search(r"[1-4]", gen_text)
        if match:
            answer = int(match.group(0))
        else:
            answer = 1
        answers.append(answer)
    return answers


def evaluate_quiz(quiz_path, score_path=None, results_path=None, batch_size=1):
    """Evaluate a quiz JSON file and save results. Uses batching for speed."""

    with open(quiz_path, 'r') as f:
        quiz = json.load(f)

    # Trim to first_n questions for quick testing
    if hasattr(evaluate_quiz, 'first_n') and evaluate_quiz.first_n is not None:
        quiz = quiz[:evaluate_quiz.first_n]

    if results_path is None:
        results_path = quiz_path.replace('.json', '_model_answers.json')
    if score_path is None:
        score_path = 'quiz_score.txt'

    results = []
    correct = 0
    n = len(quiz)
    for start in tqdm(range(0, n, batch_size), desc="Evaluating quiz (batched)"):
        batch = quiz[start:start+batch_size]
        options_list = [q['options'] for q in batch]
        answers = choose_outlier_batch(options_list)
        for q, answer in zip(batch, answers):
            q_id = q['question_id']
            concept = q['concept']
            correct_answer = q.get('correct_answer', None)
            if correct_answer is not None:
                correct_answer = correct_answer + 1  # Convert to 1-indexed
            is_correct = (answer == correct_answer) if correct_answer else None
            if is_correct:
                correct += 1
            results.append({
                'question_id': q_id,
                'concept': concept,
                'model_answer': answer,
                'correct_answer': correct_answer,
                'correct': is_correct
            })

    accuracy = correct / len(quiz) if any(r['correct'] is not None for r in results) else 0
    print(f"\nQuiz score: {correct}/{len(quiz)} ({accuracy*100:.1f}%)")


    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save score to file
    score_dir = os.path.dirname(score_path)
    if score_dir:
        os.makedirs(score_dir, exist_ok=True)
    with open(score_path, 'w') as f:
        f.write(f"Quiz score: {correct}/{len(quiz)} ({accuracy*100:.1f}%)\n")

    print(f"\nQuiz score: {correct}/{len(quiz)} ({accuracy*100:.1f}%)")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a quiz JSON file.")
    parser.add_argument("--quiz_path", default="output/quiz_with_answers.json", help="Path to the quiz JSON file.")
    parser.add_argument("--score_path", default="quiz_score.txt", help="Path to save the evaluation results.")
    parser.add_argument("--results_path", default="output/quiz_model_answers.json", help="Path to save the model's answers.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for model inference.")
    parser.add_argument("--first_n", type=int, default=None, help="Only evaluate the first N questions (for quick testing).")
    args = parser.parse_args()
    # Pass first_n via function attribute for access inside evaluate_quiz
    evaluate_quiz.first_n = args.first_n
    evaluate_quiz(args.quiz_path, score_path=args.score_path, results_path=args.results_path, batch_size=args.batch_size)