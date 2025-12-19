from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch
from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer
from datasets import load_dataset

train_dataset = load_dataset(
    "./data/llava-instruct-mix",
    split="train[:10%]",
)

model_name = "models/Qwen3-VL-8B-Instruct" 

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
        bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
        bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
        bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
    )
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
)

output_dir = "output/models/Qwen3-VL-8B-Instruct-sft"

training_args = SFTConfig(
    #num_train_epochs=1,
    # max_steps=10,                                         
    per_device_train_batch_size=2,                        
    gradient_accumulation_steps=8,                        # effective batch size = 4 * 8 = 32
    warmup_steps=5,                                       
    learning_rate=2e-4,                                   
    optim="adamw_8bit",                                   
    max_length=None,                                      # truncating may remove image tokens

    output_dir=output_dir,                                
    logging_steps=1,                                     
    report_to="trackio",                                  

    # push_to_hub=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
trainer.save_model(output_dir)
