import torch
import argparse
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from swift import Swift

def main(args):
    # default: Load the model on the available device(s)--model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_base,  torch_dtype="auto", device_map=None, tp_plan=None
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(args.model_base)
    # Load LoRA weight
    # model = PeftModel.from_pretrained(model, args.model_path)
    model = Swift.from_pretrained(model, args.model_path)


    # merge weight
    model = model.merge_and_unload()
    # print(model)
    model.save_pretrained(args.model_save)
    processor.save_pretrained(args.model_save)

    print("<<<DONE>>>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--model-save", type=str)
    args = parser.parse_args()
    # python merge_lora_swift.py --model-path ckpt/critic_exp11_massive/v0-20250612-103251/checkpoint-9500 --model-base Qwen/Qwen2.5-VL-7B-Instruct --model-save ckpt/critic_exp11_massive/v0-20250612-103251/checkpoint-merged
    main(args)

