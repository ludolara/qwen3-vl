import os

import torch
from datasets import Features, Image, List, Value, load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _is_main_process() -> bool:
    return _get_env_int("RANK", 0) == 0


def _load_train_dataset():
    features = Features(
        {
            "images": List(Image()),
            "prompt": List({"role": Value("string"), "content": Value("string")}),
            "completion": List({"role": Value("string"), "content": Value("string")}),
        }
    )
    cache_dir = os.environ.get("HF_DATASETS_CACHE", "output/hf_cache")
    return load_dataset(
        # "./data/highlighted_images_v4.5_hf_copy",
        "./data/llava-instruct-mix",
        split="train[:1%]",
        # split="train",
    )


def main() -> None:
    local_rank = _get_env_int("LOCAL_RANK", -1)
    world_size = _get_env_int("WORLD_SIZE", 1)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit multi-GPU training.")

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = {"": 0}

    train_dataset = _load_train_dataset()
    if _is_main_process():
        print(f"Train samples: {len(train_dataset)}")

    model_name = "models/Qwen3-VL-2B-Instruct"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "down_proj",
            "o_proj",
            "k_proj",
            "q_proj",
            "gate_proj",
            "up_proj",
            "v_proj",
        ],
    )

    output_dir = "output/models/Qwen3-VL-2B-Instruct-sft"

    training_args = SFTConfig(
        # num_train_epochs=1,
        max_steps=5,
        per_device_train_batch_size=4,
        # gradient_accumulation_steps=8,
        # warmup_steps=5,
        learning_rate=2e-4,
        optim="adamw_8bit",
        max_length=None,
        output_dir=output_dir,
        logging_steps=1,
        report_to="none",
        ddp_find_unused_parameters=False if world_size > 1 else None,
        local_rank=local_rank,
    )

    processor = AutoProcessor.from_pretrained(model_name)
    data_collator = DataCollatorForVisionLanguageModeling(processor=processor)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=data_collator,
    )

    if _is_main_process():
        gpu_stats = torch.cuda.get_device_properties(torch.cuda.current_device())
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
