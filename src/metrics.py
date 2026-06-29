import sys
import json

def calculate_f1(file_path):
    tp = fp = fn = tn = 0  # 不需要 tn，因为不参与 F1 计算

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            label = data.get("label")
            predicted = data.get("predicted")

            # 假设 "correct" 是正类，"wrong" 是负类
            if label == "correct" and predicted == "correct":
                tp += 1
            elif label == "wrong" and predicted == "correct":
                fp += 1
            elif label == "correct":
                fn += 1  # predicted is "wrong" or unparsed (null) -> a missed positive
            else:
                tn += 1

    # 计算 Precision、Recall、F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print(f"\nTP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return f1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python calculate_f1_from_jsonl.py <文件路径>")
    else:
        calculate_f1(sys.argv[1])