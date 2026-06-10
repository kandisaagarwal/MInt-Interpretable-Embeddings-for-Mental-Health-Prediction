#!/usr/bin/env python3

"""Score a quiz by choosing the longest or shortest option text."""

import argparse
import json
from pathlib import Path


DEFAULT_QUIZ_PATH = "output_mentalbert_1/mental_bert_quiz.json"


def choose_by_length(options, mode):
    lengths = [len(str(option)) for option in options]
    if mode == "largest":
        target_length = max(lengths)
        for index, length in enumerate(lengths):
            if length == target_length:
                return index
    elif mode == "smallest":
        target_length = min(lengths)
        for index, length in enumerate(lengths):
            if length == target_length:
                return index
    elif mode == "most_different":
        scores = []
        for index, length in enumerate(lengths):
            score = sum(abs(length - other_length) for other_length in lengths if other_length != length or lengths.count(length) == 1)
            scores.append(score)
        return max(range(len(scores)), key=lambda index: scores[index])
    return 0


def score_quiz(quiz, mode):
    correct = 0
    predictions = []
    for question in quiz:
        predicted_answer = choose_by_length(question["options"], mode)
        gold_answer = question.get("correct_answer")
        is_correct = gold_answer is not None and predicted_answer == gold_answer
        if is_correct:
            correct += 1
        predictions.append(
            {
                "question_id": question.get("question_id"),
                "predicted_answer": predicted_answer,
                "correct_answer": gold_answer,
                "correct": is_correct,
            }
        )

    total = len(quiz)
    accuracy = (correct / total) if total else 0.0
    return correct, total, accuracy, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Score a quiz by picking the longest or shortest option."
    )
    parser.add_argument(
        "--quiz_path",
        default=DEFAULT_QUIZ_PATH,
        help="Path to the quiz JSON file with correct answers.",
    )
    parser.add_argument(
        "--results_path",
        default=None,
        help="Optional path to save the per-question predictions as JSON.",
    )
    args = parser.parse_args()

    quiz_path = Path(args.quiz_path)
    with quiz_path.open("r", encoding="utf-8") as handle:
        quiz = json.load(handle)

    all_results = {}
    for mode in ("largest", "smallest", "most_different"):
        correct, total, accuracy, predictions = score_quiz(quiz, mode)
        all_results[mode] = predictions
        label = "Most different" if mode == "most_different" else mode.title()
        print(f"{label} option score: {correct}/{total} ({accuracy * 100:.1f}%)")

    if args.results_path:
        results_path = Path(args.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)


if __name__ == "__main__":
    main()