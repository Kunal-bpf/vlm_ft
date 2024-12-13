import os
import json
import torch
import plotly.graph_objects as go
import logging

from logging.handlers import RotatingFileHandler
from unsloth import FastVisionModel
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from PIL import Image


# Create a logger
logger = logging.getLogger("DirectLogger")
logger.setLevel(logging.DEBUG)

# Formatter for logs
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# File handler with rotation
file_handler = RotatingFileHandler("model_training.log", maxBytes=10 * 1024 * 1024, backupCount=3)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logger.info)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Load the model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Configure fine-tuning parameters
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=128,
    lora_alpha=128,
    lora_dropout=0.1,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

instruction = """You are a Vision Language Model specialized in Optical Character Recognition (OCR) for Indian vehicle number plates. Your task is to extract and return the exact text from an image of a number plate, ensuring the following guidelines are strictly followed:

Focus: Analyze only the region of the number plate and extract the alphanumeric text displayed on it.

Character Handling:

Indian number plates typically contain combinations of uppercase English alphabets (A-Z) and numbers (0-9).
If a character is partially visible, blurry, or unidentifiable, replace it with a question mark (?).
Formatting:

Return the text exactly as it appears on the number plate (e.g., "MH12AB1234").
Do not add any extra characters, spaces, or descriptions.
Ensure no surrounding context, metadata, or additional text is included in the output.
Special Note for Variants:

Some Indian plates may contain state codes, district codes, and vehicle registration numbers in various alignments.
Handle non-standard formatting gracefully, ensuring only the text on the plate is extracted.
Output: Provide a JSON BLOCK that represents the number plate. Example outputs:

```json{
    "license_number": "KA03MP1234"
}```

```json{
    "license_number": "DL5CAC0001"
}```

```json{
    "license_number": "AP09CD5678"
}```

```json{
    "license_number": "MH??XY789?"
}```

Input: An image containing an Indian vehicle number plate.

Output: A JSON BLOCK representing the extracted number plate with ? for unidentifiable characters."""


def data_prep(split, set_path):
    # Get a list of all files in the directory
    filenames = os.listdir(set_path)

    # Prepare the paths and filenames
    paths_with_filename = {
        "path": [os.path.abspath(os.path.join(set_path, fname)) for fname in filenames],
        "filename": [fname for fname in filenames],
    }

    output_file = f"{split}_data.jsonl"
    with open(output_file, "w") as file:
        pass

    for i in range(len(paths_with_filename["path"])):
        full_path = paths_with_filename["path"][i]
        filename = paths_with_filename["filename"][i]

        if filename.lower().endswith(".jpg"):
            clean_filename = filename.removesuffix(".jpg")
            item = {
                "path": full_path,
                "filename": clean_filename.split("_")[0],
            }

            with open(output_file, "a") as file:
                file.write(json.dumps(item) + "\n")

    logger.info(f"Data preparation complete. JSONL file saved to '{output_file}'.")


train_path = "./ANPR_Dataset/crops/train_license_plate_crops"
val_path = "./ANPR_Dataset/crops/val_license_plate_crops"

data_prep("train", train_path)
data_prep("val", val_path)


with open("train_data.jsonl", "r") as file:
    train_data = [json.loads(line) for line in file]

with open("val_data.jsonl", "r") as file:
    val_data = [json.loads(line) for line in file]


def format_to_json(filename):
    return f"""```json
{{
    "license_number": "{filename}"
}}
```"""


def get_image(path):
    try:
        image = Image.open(path)
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


def convert_to_conversation(sample):
    image = get_image(sample["path"])

    if image is None:
        return None

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": format_to_json(sample["filename"])}],
        },
    ]
    return {"messages": conversation}


converted_train = [
    conv
    for sample in train_data
    if (conv := convert_to_conversation(sample)) is not None
]
converted_val = [
    conv 
    for sample in val_data 
    if (conv := convert_to_conversation(sample)) is not None
]


logger.info(converted_train[0])


# Configure the trainer
FastVisionModel.for_training(model)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_train,
    eval_dataset=converted_val,
    args=SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # num_train_epochs = 5,
        do_eval=True,
        eval_strategy="steps",
        warmup_ratio=0.5,
        max_steps=150,
        learning_rate=5e-5,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=10,
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="finetuned_model_checkpoint",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=3072,
        max_grad_norm=0.3,
    ),
)

# Train the model
trainer_stats = trainer.train()


training_logs = trainer.state.log_history

train_losses = [entry["loss"] for entry in training_logs if "loss" in entry]
eval_losses = [entry["eval_loss"] for entry in training_logs if "eval_loss" in entry]

train_steps = range(1, len(train_losses) + 1)
eval_steps = range(1, len(eval_losses) + 1)


# Create a figure
fig = go.Figure()

# Add training loss line
fig.add_trace(
    go.Scatter(
        x=list(train_steps),
        y=train_losses,
        mode="lines+markers",
        name="Training Loss",
        line=dict(color="blue"),
        marker=dict(symbol="circle"),
    )
)

# Add validation loss line
fig.add_trace(
    go.Scatter(
        x=list(eval_steps),
        y=eval_losses,
        mode="lines+markers",
        name="Validation Loss",
        line=dict(color="orange"),
        marker=dict(symbol="x"),
    )
)

# Update layout with titles and labels
fig.update_layout(
    title="Training and Validation Loss (Interactive)",
    xaxis_title="Steps",
    yaxis_title="Loss",
    legend=dict(x=0, y=1),
    template="plotly_dark",  # Optional: Choose a template for style
)

fig.write_image("loss_plot_llama.png")
fig.write_html("loss_plot_llama.html")
# Show the interactive graph
# fig.show()

# saving lora adapters
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# saving merged models
model.save_pretrained_merged("merged_ft_llama", save_method="merged_16bit")
tokenizer.save_pretrained("merged_ft_llama")

# model.save_pretrained_merged("your-username/Llama-3.2-11B-Vision-Radiology-mini", tokenizer,)
# model.push_to_hub_merged("your-username/Llama-3.2-11B-Vision-Radiology-mini",
#                         tokenizer,
#                         save_method = "merged_16bit",
#                         token = "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")