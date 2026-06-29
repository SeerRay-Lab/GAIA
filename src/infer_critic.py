"""Multi-GPU inference / evaluation for the GUI action critic (ICM).

Loads a Qwen2.5-VL based critic checkpoint and, for every example in the
evaluation set, asks the critic whether the agent's candidate action is
`correct` or `wrong`. The prediction is compared against the ground-truth
label to report accuracy.

Expected input: a JSON file containing a list of records in ms-swift chat
format (see data/sample/critic_eval_sample.json):

    {
      "messages": [
        {"role": "system",    "content": "<critic system prompt>"},
        {"role": "user",      "content": "<task + action description> <image>"},
        {"role": "assistant", "content": "correct"}      # ground-truth label
      ],
      "images": ["/abs/path/to/screenshot.jpg"]
    }
"""

import os
import re
import json
import argparse
import multiprocessing as mp

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


def split_into_batches(items, batch_size):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def merge_results(output_dir, exp_name, num_gpus):
    merged_file = os.path.join(output_dir, f"{exp_name}_merged.jsonl")
    with open(merged_file, "w") as fw:
        for rank in range(num_gpus):
            part_file = os.path.join(output_dir, f"{exp_name}_gpu{rank}.jsonl")
            if not os.path.exists(part_file):
                print(f"[warn] missing shard: {part_file}")
                continue
            with open(part_file, "r") as fr:
                fw.writelines(fr)
    print(f"Merged all shards into {merged_file}")


def process_chunk(rank, args, data_chunk, stats_dict):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    critic = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="bfloat16",
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)

    out_file = os.path.join(args.output_dir, f"{args.exp_name}_gpu{rank}.jsonl")
    batches = split_into_batches(data_chunk, args.batch_size)

    matched = unmatched = correct = total = 0
    with open(out_file, "w") as f_out:
        for batch in tqdm(batches, desc=f"GPU {rank}", position=rank):
            try:
                messages, labels = [], []
                for item in batch:
                    image = item["images"][0]
                    if not os.path.exists(image):
                        continue
                    system_text = item["messages"][0]["content"]
                    user_text = item["messages"][1]["content"]
                    labels.append(item["messages"][2]["content"])
                    messages.append([
                        {"role": "system", "content": [{"type": "text", "text": system_text}]},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image", "image": image, "max_pixels": args.max_pixels},
                        ]},
                    ])

                if not messages:
                    continue

                texts = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=texts, images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(device)

                generated_ids = critic.generate(**inputs, max_new_tokens=args.max_new_tokens)
                trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
                responses = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for response, label in zip(responses, labels):
                    match = re.search(r"(correct|wrong)", response, re.DOTALL)
                    pred = match.group(1) if match else None
                    if match:
                        matched += 1
                        if str(pred) == str(label):
                            correct += 1
                    else:
                        unmatched += 1
                    f_out.write(json.dumps({
                        "response": response,
                        "predicted": pred,
                        "label": label,
                        "is_correct": pred is not None and str(pred) in str(label),
                    }, ensure_ascii=False) + "\n")
                    total += 1
            except Exception as e:  # keep going on per-batch errors
                print(f"[GPU {rank}] batch failed: {e}")
                continue

    stats_dict[f"matched_{rank}"] = matched
    stats_dict[f"unmatched_{rank}"] = unmatched
    stats_dict[f"correct_{rank}"] = correct
    stats_dict[f"total_{rank}"] = total
    print(f"[GPU {rank}] done -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="GUI critic multi-GPU inference / evaluation")
    parser.add_argument("--model_path", required=True,
                        help="Path or HF id of the critic checkpoint to evaluate")
    parser.add_argument("--input_file", required=True,
                        help="JSON file with a list of chat-format records")
    parser.add_argument("--output_dir", default="rst")
    parser.add_argument("--exp_name", default="critic_eval")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=3600 * 28 * 28)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_file, "r") as f:
        full_data = json.load(f)

    num_gpus = max(1, args.num_gpus)
    chunk = len(full_data) // num_gpus
    chunks = [full_data[i * chunk:(i + 1) * chunk] for i in range(num_gpus)]
    if len(full_data) % num_gpus:
        chunks[-1].extend(full_data[num_gpus * chunk:])

    mp.set_start_method("spawn")  # required for CUDA in subprocesses
    manager = mp.Manager()
    stats_dict = manager.dict()

    procs = []
    for rank in range(num_gpus):
        p = mp.Process(target=process_chunk, args=(rank, args, chunks[rank], stats_dict))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    merge_results(args.output_dir, args.exp_name, num_gpus)

    total_matched = sum(stats_dict[f"matched_{r}"] for r in range(num_gpus))
    total_correct = sum(stats_dict[f"correct_{r}"] for r in range(num_gpus))
    total_entries = sum(stats_dict[f"total_{r}"] for r in range(num_gpus))

    print("\n--- Summary ---")
    print(f"Total entries:      {total_entries}")
    print(f"Matched (parsed):   {total_matched}")
    print(f"Correct:            {total_correct}")
    if total_matched:
        print(f"Accuracy (matched): {total_correct / total_matched * 100:.2f}%")
    else:
        print("No parsable predictions.")


if __name__ == "__main__":
    main()
