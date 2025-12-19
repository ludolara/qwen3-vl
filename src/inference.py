from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datasets import load_dataset

train_dataset = load_dataset(
    "data/llava-instruct-mix",
    split="train[:10%]",
)

base_model = "models/Qwen3-VL-8B-Instruct"
adapter_model = f"output/models/Qwen3-VL-8B-Instruct-sft"

model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_model)

processor = AutoProcessor.from_pretrained(base_model)

problem = train_dataset[0]['prompt'][0]['content']
image = train_dataset[0]['images'][0]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": problem},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
