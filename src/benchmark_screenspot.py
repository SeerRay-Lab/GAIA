from transformers import AutoProcessor, AutoModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
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
    messages = [
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
     # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    width, height = image_inputs[0].size
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample=True, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(output_text)
    output = json.loads(output_text.replace("(","[").replace(")","]"))[0]
    position = output['point_2d']
    x, y = position
    abs_x = x/input_width*width
    abs_y = y/input_height*height
    
    return [abs_x, abs_y]

def eval_model_txt(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
            instruction = "Locate the position of the instruction \"{}\" and output the coordinate in JSON format.".format(item["instruction"])
            #print(instruction)
            q = item["instruction"]
            gt_bbox = item["bbox"]
            gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
            for i in range(1): 
                try:
                    output = single_conv(img_path, instruction, model, processor)
                    x1, y1 =output
                except:
                    x1,y1=0,0 # for point
                #print(x1, y1)
            
            try:
            #if True:
                #ori_box = [width*(x1/1000), height*(y1/1000), width*(x2/1000), height*(y2/1000)]
                #click_point = [(ori_box[0] + ori_box[2]) / 2, (ori_box[1] + ori_box[3]) / 2]
                
                #click_point = [width*(x1/1000), height*(y1/1000)]
                click_point = [x1, y1]
                #print("click_point: ",click_point, " gt_bbox: ",gt_bbox)
                if (gt_bbox[0] <= click_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= click_point[1] <= gt_bbox[3]):
                    corr_action += 1
                    if item["data_type"] == 'text':
                        text_correct.append(1)
                    else:
                        icon_correct.append(1)
                    #logging.info("match " + str(corr_action / num_action))
                    total_true += 1
                else:
                        print(click_point, gt_bbox)
                        if item["data_type"] == 'text':
                            text_correct.append(0)
                        else:
                            icon_correct.append(0)
                        logging.info("unmatch " + str(corr_action / num_action))
                        total_false += 1
                        cv2.circle(image_show, (int(click_point[0]), int(click_point[1])),10,(0,0,255), 5)
                        #cv2.rectangle(image_show, (int(ori_box[0]), int(ori_box[1])),(int(ori_box[2]), int(ori_box[3])), (0,0,255),2)
                        cv2.rectangle(image_show, (gt_bbox[0], gt_bbox[1]),(gt_bbox[2], gt_bbox[3]), (0,255,0),2)
                        cv2.imwrite("./badcase/"+q+"-"+filename, image_show)


            except:
                num_wrong_format += 1
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("Step: " + str(j) + " wrong format")
                total_false += 1

            total_pred += 1
            #logging.info("total: "+str(total_pred)+" true: "+str(total_true)+" false: "+str(total_false)+" perf: "+str(total_true / total_pred))

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
    parser.add_argument("--task", type=str, default="all")

    args = parser.parse_args()
    eval_model_txt(args)
