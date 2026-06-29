from transformers import AutoProcessor, AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import set_seed
import argparse
from tqdm import tqdm
import os
import json
import torch
from PIL import Image
import re
import logging
import cv2
from qwen_vl_utils import process_vision_info

logging.basicConfig(level = logging.INFO)



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
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample=True, temperature=1.0, top_k=30, top_p=0.8, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts

def eval_model_txt(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        )

    # default processer
    #processor = AutoProcessor.from_pretrained(args.model_path)
    min_pixels = 256*28*28
    max_pixels = 2700*28*28
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
            num_action += 1
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)
            image = Image.open(img_path).convert("RGB")
            image_show = cv2.imread(img_path)
            width, height = image.size
            #instruction = "Click to "+item["instruction"]
            input_instruction = json.dumps({"thought": item["instruction"]}, ensure_ascii=False)

            #print(instruction)
            q = item["instruction"]
            gt_bbox = item["bbox"]
            gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
            
            correct_flag = False

            outputs = single_conv(img_path, input_instruction, model, processor)
            for output in outputs:
                try:
                    output = json.loads(output)
                    if output['func'] == 'Tap':
                        position = output['position']
                        x1, y1 =position
                    else:
                        x1, y1 = 0,0
                except Exception as e:
                    x1, y1 = 0,0
                    print(e)
                click_point = [width*x1, height*y1]
                #print("click_point: ",click_point, " gt_bbox: ",gt_bbox)
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
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument("--screenspot_imgs", type=str, default="data/ScreenSpot-v2/screenspotv2_image/")
    parser.add_argument("--screenspot_test", type=str, default="data/ScreenSpot-v2/")
    parser.add_argument("--task", type=str, default="mobile")

    args = parser.parse_args()
    eval_model_txt(args)
