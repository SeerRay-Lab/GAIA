from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from collections import defaultdict
import sys
import re
import os
from tqdm import tqdm
import json
import cv2
import time
import base64
import numpy as np
import multiprocessing as mp
import glob
import random
from PIL import Image
from io import BytesIO
import base64
import math

# SFT critic system prompt — matches the released dataset (prompts/critic_sft.txt).
# (The <link> prompt in prompts/gui_critic.txt is the separate GRPO-stage prompt.)
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "critic_sft.txt")
with open(_PROMPT_PATH, encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()
origin_height = 2400   # fallback only, used if an image cannot be opened
origin_width = 1080
MAX_PIXELS = 3600*28*28
SIMPLE_OR_THINK = 0 # 0 for simple, 1 for think


# Qwen3 summarizer — loaded lazily, only when thinking-style labels are requested.
QWEN3_MODEL = "Qwen/Qwen3-30B-A3B"
_tokenizer = None
_model = None

def _load_qwen3():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
    return _tokenizer, _model

def generate_description_from_thought_qwen3(thought_text):
    if not thought_text:
        return ""
    tokenizer, model = _load_qwen3()

    # prepare the model input
    prompt = "你是一个擅长总结的GUI Agent。接下来我将给你发送一段话，这段话描述了GUI Agent任务过程中的一些思考信息，这些思考基于对历史轨迹、当前状态的观察和任务理解，进而评判了当前所选的动作是否符合预期，能够支持任务完成。但是这个思考内容太长了，请帮我总结成一两句话。注意避开原句中有关'正确操作'的GT信息。文本如下:{}\n".format(thought_text)
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=132768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by "factor".
    2. The total number of pixels is within the range ["min_pixels", "max_pixels"].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def process_jsonl_to_jsonl(input_paths, output_path, idx, n_parallel):
    try:
        print(f"\n=== 开始处理文件 ===")
        print(f"输入路径: {input_paths}")
        print(f"输出路径: {output_path}")
        print(f"处理第 {idx+1}/{n_parallel} 段数据")

        lines = []
        # 读取JSONL文件
        for path in input_paths:
            # 获取不含父目录的文件名
            filename = os.path.basename(path)

            # 判断是 good 还是 bad
            quality = None
            if 'good' in filename:
                quality = 'good'
            elif 'bad' in filename:
                quality = 'bad'
            else:
                print(f"文件 {filename} 不含 'good' 或 'bad' 标识")
            
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    try:
                        record = json.loads(stripped_line)
                        # 检查images字段是否已存在
                        if 'image' in record:
                            image_path = record['image']
                            if quality is not None:
                                record['case'] = quality
                            lines.append(record)
                            # if image_path not in existing_images:
                            #     lines.append(record)
                                # existing_images.add(image_path)
                    except json.JSONDecodeError as e:
                        print(f"文件 {path} 第 {line_num + 1} 行解析失败: {e}")
        
        # 计算当前段的数据范围
        total_lines = len(lines)
        chunk_size = math.ceil(total_lines / n_parallel)
        start_idx = idx * chunk_size
        end_idx = min((idx + 1) * chunk_size, total_lines)
        current_chunk = lines[start_idx:end_idx]
        
        print(f"总数据量: {total_lines}")
        print(f"当前处理段: {start_idx} - {end_idx}")
        
        # 初始化状态管理
        request_steps = defaultdict(int)
        processed_data = []
        sft_processed_data = []

        # 分析字段
        field_counter = defaultdict(int)
        example_values = {}

        for record in current_chunk:
            for key in record.keys():
                field_counter[key] += 1
                if key not in example_values and record[key]:
                    example_values[key] = repr(record[key])[:100]

        # 打印字段分析
        print("\n=== 输入字段分析 ===")
        print("字段名 | 出现次数 | 示例值")
        print("-" * 60)
        for field in sorted(field_counter):
            count = field_counter[field]
            example = example_values.get(field, '')
            print(f"{field:<20} | {count:<5} | {example}")

        # 处理每条记录
        save_interval = 100  # 每处理100条数据保存一次
        temp_sft_data = []  # 临时存储处理后的数据
        
        for idx, record in tqdm(enumerate(current_chunk), total=len(current_chunk), desc=f"处理数据记录(第{idx+1}段)"):
            request_id = record.get('id')
            traj_id = request_id.split('_')[0]
            step_id = request_id.split('_')[1]

            user_text = record.get('instruction', '')
            history = record.get('action_history', '')
            thought = record.get('pred_action', '')
            image = record.get('image', '')

            # Rescale tap coordinates using the ACTUAL screenshot size (falls back
            # to the defaults only if the image cannot be opened).
            action = thought
            img_w, img_h = origin_width, origin_height
            if image:
                try:
                    img_w, img_h = Image.open(image).size
                except Exception:
                    img_w, img_h = origin_width, origin_height
            if isinstance(action, dict) and 'x' in action and 'y' in action:
                resized_h, resized_w = smart_resize(height=img_h, width=img_w, max_pixels=MAX_PIXELS)
                action['x'] = action['x'] / img_w * resized_w
                action['y'] = action['y'] / img_h * resized_h

            # Build the user message in the released dataset format.
            text_content = f"The goal of the task (instruction): {user_text}\n"
            text_content += f"Action (plan) history: {history}\n"
            text_content += f"Current action of the agent: {action}\n"
            text_content += "Screenshot: "

            # for simple
            if SIMPLE_OR_THINK == 0:
                if record.get('case', '') == 'good':
                    label = 'correct'
                elif record.get('case', '') == 'bad':
                    label = 'wrong'
            # for think
            elif SIMPLE_OR_THINK == 1:
                review_text = record.get('review', '')
                match = re.search(r'<Tag>(.*?)</Tag>', review_text, re.DOTALL)
                tag = match.group(1).strip() if match else ''
                case_judge = record.get('case', '')
                if case_judge == 'good' and 'Wrong' in tag:
                    continue
                if case_judge == 'bad' and 'Correct' in tag:
                    continue
                # prepend a short rationale before the verdict
                verdict = 'correct' if case_judge == 'good' else 'wrong'
                description = generate_description_from_thought_qwen3(review_text)
                label = f"{description}\n{verdict}"

            # 构建新记录
            new_record = {
                'request_id': traj_id,
                'step_id': step_id,
                'user_text': user_text,
                'history': history,
                'thought': thought,
                'image': image,
                'label': label
            }
            processed_data.append(new_record)

            new_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": text_content+'<image>'
                    },
                    {
                        "role": "assistant",
                        "content": label
                    }
                ],
                "images": [new_record['image']] if new_record['image'] else []
            }
            temp_sft_data.append(new_item)
            sft_processed_data.append(new_item)

            request_steps[traj_id] += 1

            # 每处理save_interval条数据保存一次
            # if (idx + 1) % save_interval == 0:
            #     temp_output_path = f"{output_path}.temp_{idx + 1}"
            #     print(f"\n保存临时文件: {temp_output_path}")
            #     with open(temp_output_path, 'w', encoding='utf-8') as f:
            #         json.dump(temp_sft_data, f, ensure_ascii=False, indent=2)
                # temp_sft_data = []  # 清空临时数据

        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sft_processed_data, f, ensure_ascii=False, indent=2)

        # 统计信息
        unique_requests = len(request_steps)
        total_steps = sum(request_steps.values())

        print("\n=== 处理统计 ===")
        print(f"唯一request_id数量: {unique_requests}")
        print(f"总步骤数: {total_steps}")
        print(f"平均每个request的步骤数: {total_steps / unique_requests:.2f}")

        # 标签分布统计
        label_counts = {}
        for item in processed_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        # print("\n=== 标签分布 ===")
        # for label, count in label_counts.items():
        #     percentage = (count / len(processed_data)) * 100
        #     print(f"标签 '{label}': {count} 条 ({percentage:.2f}%)")

        # 示例输出
        if processed_data:
            print("\n=== 数据样例 ===")
            print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))

        return True

    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        return False


def main():
    """命令行接口"""
    # source path/to/venv/bin/activate
    # cd .
    # CUDA_VISIBLE_DEVICES=0 python generate_sft_dataset_from_good_bad_cases.py badcase_qwen2_5_vl_32b_review_merged.jsonl goodcase_qwen2_5_vl_32b_review_merged.jsonl sft_data_thinking_qwen3_android_control.json 0 8
    # python src/generate_sft_dataset_from_good_bad_cases.py data_raw/inference_good_result.jsonl data_raw/inference_bad_result.jsonl data/sft_data_simple_android_control_val.json 0 1
    if len(sys.argv) < 5:
        print("用法: python script.py <输入文件1> [输入文件2 ...] <输出文件路径> <段索引> <并行度>")
        return

    input_files = sys.argv[1:-3]
    output_file = sys.argv[-3]
    idx = int(sys.argv[-2])
    n_parallel = int(sys.argv[-1])

    # 检查输入文件是否存在
    for file in input_files:
        if not os.path.isfile(file):
            print(f"错误：输入文件 {file} 不存在")
            return

    # Only add a part suffix when sharding across multiple parallel workers.
    if n_parallel > 1:
        base_path, ext = os.path.splitext(output_file)
        output_file = f"{base_path}_part{idx}{ext}"

    print(f"开始处理文件: {input_files}")
    print(f"处理第 {idx+1}/{n_parallel} 段数据")
    success = process_jsonl_to_jsonl(input_files, output_file, idx, n_parallel)

    if success:
        print(f"\n\n处理完成，结果保存至: {output_file}")
    else:
        print("\n\n处理失败，请检查日志")


if __name__ == "__main__":
    main()