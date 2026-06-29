from transformers import AutoProcessor, AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import set_seed
import argparse
from tqdm import tqdm
import os
import json
import torch
import math
from PIL import Image
import re
import logging
import cv2
from qwen_vl_utils import process_vision_info

logging.basicConfig(level = logging.INFO)


# Critic system prompt — the prompt the released critic was trained with.
_CRITIC_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "gui_critic.txt")
with open(_CRITIC_PROMPT_PATH, encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()

USER_PROMPT = "请根据\n1.整体任务\n2.当前屏幕截图\n3.历史操作\n4.需要在当前页面执行的操作。判断当前页面执行的操作是否正确。整体任务:{}\n屏幕截图:"
USER_PROMPT2 = "\n历史操作:\n在当前页面执行的操作:\"action_type\": {}, \"x\": {}, \"y\": {}"

# USER_PROMPT = "请根据\n1.整体任务\n2.当前屏幕截图\n3.历史操作\n4.需要在当前页面执行的操作。判断当前页面执行的操作是否正确。整体任务:{}\n屏幕截图:"
# USER_PROMPT2 = "\n历史操作:{}\n在当前页面执行的操作:{}"
# {"action_type": "click", "x": 327, "y": 884}

actor_device = "cuda:0"
critic_device = "cuda:0"

min_pixels = 256*28*28
max_pixels = 3600*28*28
critic_model = None
critic_processor = None

def load_critic(critic_path):
    """Load the critic model/processor once (call after argparse)."""
    global critic_model, critic_processor
    critic_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        critic_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=critic_device,
    )
    critic_processor = AutoProcessor.from_pretrained(
        critic_path, min_pixels=min_pixels, max_pixels=max_pixels
    )


MAX_PIXELS = 3600*28*28
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


def single_critic(image_path, query, actor_output):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(height = height, width = width, max_pixels=MAX_PIXELS)
    messages = []
    for cur_actor in actor_output:
        try:
            actor_data = json.loads(cur_actor)
            position = actor_data.get('position')
            func = actor_data.get('func')
        except Exception:
            position = None
            func = None
        if position and len(position) >= 2:
            # abs_x = position[0] * width
            # abs_y = position[1] * height
            resized_x = int(position[0] * resized_width)
            resized_y = int(position[1] * resized_height)
        else:
            resized_x = resized_y = None
        message = [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": USER_PROMPT.format(json.loads(query)["thought"])
                    },
                    {
                        "type": "image", 
                        "image":image_path
                    },
                    {
                        "type": "text", 
                        "text": USER_PROMPT2.format(func, resized_x, resized_y)  # {"action_type": "click", "x": 327, "y": 884}
                    }
                ]
            }
        ]
        messages.append(message)
    texts = [
        critic_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    # print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = critic_processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14
    inputs = inputs.to(critic_device)
    generated_ids = critic_model.generate(**inputs, do_sample=False, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    responses = critic_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return responses


def single_conv(image_path, query, model, processor):
    message = [
        {
            "role": "system",
            "content": "你是个可以分析手机屏幕的AI视觉助手，接下来你会收到一个来自用户的点击手机屏幕上的某个UI元素的thought，你需要根据收到的thought检测并定位符合描述的UI元素的位置，这个UI元素可能是文本也可能是屏幕上的某个图案。检测并定位后你需要按照json格式输出func以及position,text,app,direction等func相关的参数。"
        },

        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                "type": "text",
                "text": query
                },
            ],
        }
    ]
    messages = []
    for i in range(8):
        messages.append(message)
    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    # print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    width, height = image_inputs[0].size
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14
    inputs = inputs.to(actor_device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample=True, temperature=1.0, top_k=30, top_p=0.8, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # call critic eval
    critic_outputs = single_critic(image_path, query, output_texts)

    return output_texts, critic_outputs

def eval_model_txt(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=actor_device,
        )

    # default processer
    #processor = AutoProcessor.from_pretrained(args.model_path)
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    if args.task == "all":
        tasks = ["mobile", "desktop", "web"]
    else:
        tasks = [args.task]
    tasks_result = []

    total_pred = 0
    total_true = 0
    total_false = 0

    for task in tasks:
        dataset = "screenspot_" + task + "_v2.json"
        screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
        num_action = 0
        corr_action = 0
        text_correct = []
        icon_correct = []
        num_wrong_format = 0
        for j, item in tqdm(enumerate(screenspot_data)):
            # print("-"*50)
            num_action += 1
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)
            image = Image.open(img_path).convert("RGB")
            image_show = cv2.imread(img_path)
            width, height = image.size
            #instruction = "Click to "+item["instruction"]
            input_instruction = json.dumps({"thought": item["instruction"]}, ensure_ascii=False)

            # print(instruction)
            q = item["instruction"]
            # print(q)
            gt_bbox = item["bbox"]
            gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
            
            correct_flag = False
            outputs, critic_outputs = single_conv(img_path, input_instruction, model, processor)

            # for output in outputs:
            #     # print(output)
            #     try:
            #         output = json.loads(output)
            #         if output['func'] == 'Tap':
            #             position = output['position']
            #             x1, y1 = position
            #         else:
            #             x1, y1 = 0,0
            #     except Exception as e:
            #         x1, y1 = 0,0
            #         # print(e)
            #     click_point = [width*x1, height*y1]
            #     # print("click_point: ", click_point, " gt_bbox: ", gt_bbox)
            #     if (gt_bbox[0] <= click_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= click_point[1] <= gt_bbox[3]):
            #         # print("HERE!")
            #         correct_flag = True
            # for output in critic_outputs:
            #     # print(output)
            #     match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
            #     if match:
            #         content = match.group(1)
            #         # print(content)
            #     else:
            #         print("no match")

            
            #selected_index = None
            selected_index = 0
            for idx, output in enumerate(critic_outputs):
                link = re.search(r"<link>\s*(correct|wrong)\s*</link>", output, re.IGNORECASE | re.DOTALL)
                verdict = link.group(1).lower() if link else None
                if verdict is None:
                    fallback = re.search(r"\b(correct|wrong)\b", output, re.IGNORECASE)
                    verdict = fallback.group(1).lower() if fallback else None
                if verdict == "correct":
                    selected_index = idx
                    break
            
            if selected_index is not None:
                try:
                    output = json.loads(outputs[selected_index])
                    if output['func'] == 'Tap':
                        position = output['position']
                        x1, y1 = position
                    else:
                        x1, y1 = 0, 0
                except Exception as e:
                    x1, y1 = 0, 0
                # print(output)
                click_point = [width * x1, height * y1]
                if (gt_bbox[0] <= click_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= click_point[1] <= gt_bbox[3]):
                    correct_flag = True
            
            if correct_flag:
                corr_action += 1
                if item["data_type"] == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                total_true += 1
            else:
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("unmatch " + str(corr_action / num_action))
                total_false += 1
            
            total_pred += 1

            #if total_pred == 5:
            #    break

        logging.info("Action Acc: " + str(corr_action / num_action))
        logging.info("Total num: " + str(num_action))
        logging.info("Wrong format num: " + str(num_wrong_format))
        logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
        logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))

        text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
        icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
        tasks_result.append([text_acc, icon_acc])

    logging.info(tasks_result)
    logging.info("total: "+str(total_pred)+" true: "+str(total_true)+" false: "+str(total_false)+" perf: "+str(total_true / total_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/actor', help='actor (policy) model')
    parser.add_argument('--critic_path', type=str, default='checkpoints/critic', help='trained critic checkpoint')
    parser.add_argument("--screenspot_imgs", type=str, default="data/ScreenSpot-v2/screenspotv2_image/")
    parser.add_argument("--screenspot_test", type=str, default="data/ScreenSpot-v2/")
    parser.add_argument("--task", type=str, default="mobile")

    args = parser.parse_args()
    load_critic(args.critic_path)
    eval_model_txt(args)
