instruction ="""You are a Vision Language Model specialized in Optical Character Recognition (OCR) for Indian vehicle number plates. Your task is to extract and return the exact text from an image of a number plate, ensuring the following guidelines are strictly followed:

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


from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image

model_name = "./merged_ft_llama"

llm = LLM(
    model=model_name,
    tokenizer="./merged_ft_llama",
    tokenizer_mode="auto",
    max_model_len=3072,
    max_num_seqs=16,
    enforce_eager=True,
    gpu_memory_utilization=0.5,
)
sampling_params = SamplingParams(temperature=1e-12,
                                    max_tokens=100,
                                    stop=["<|eot_id|>"],
                                    top_k=1
                                )

tokenizer = AutoTokenizer.from_pretrained("./merged_ft_llama")

def get_response(image_path):

    modality = "image"
    try:
        image = Image.open(image_path)
        # return image

        messages = [
            { "role" : "user", "content": [{"type" : "text", "text" : instruction}]}
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize=False)

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: image
            },
        }

        outputs = llm.generate(inputs, sampling_params=sampling_params)
        # print(outputs[0].outputs[0].text)
        return outputs[0].outputs[0].text
    
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
import re
import json
import time

def parse_json(response):
    try:
        if response is not None:
            pattern = r'```json(.*?)```'

            match = re.search(pattern, response, flags=re.DOTALL)
            json_data = match.group(1)

            data = json.loads(json_data)

            return data
        else:
            return None
    except Exception as e:
        print(e)
        return None
    
with open("val_data.jsonl",'r') as file:
    val_data = [json.loads(line) for line in file]

accuracy = 0
size = len(val_data)
y = []
y_pred = []
pred_time = []


for i in range(size):
    path = val_data[i]['path']
    filename = val_data[i]['filename'].split("_")[0]
    y.append(filename)
    
    start = time.time()
    json_data = parse_json(get_response(path))
    end = time.time()

    pred_time.append(end-start)

    if json_data is None:
        y_pred.append("NaN")
    else:
        y_pred.append(json_data['license_number'])

        if json_data['license_number'] == filename:
            accuracy += 1

print(f"The accuracy on {size} sample is: {(accuracy/size) * 100}")

import pandas as pd

df = pd.DataFrame({"y": y, "y_pred": y_pred, "pred_time" : pred_time})
df["isa_match"] = [1 if y_val == y_pred_val else 0 for y_val, y_pred_val in zip(df["y"], df["y_pred"])]

df.to_csv("results.csv", index=False)