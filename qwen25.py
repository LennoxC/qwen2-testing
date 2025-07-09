from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from torch import float16

def doqwen25(imagepath, prompt):
    # Load the model and processor
    model = AutoModelForVision2Seq.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=float16, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)

    #image = Image.open(f"./testImage/{imagename}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": imagepath
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cpu")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)