# ref : https://huggingface.co/MILVLG/imp-v1-3b

# pip install transformers pillow accelerate einops --user

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# note - disable this if your system does not have nvidia cuda hardware and software
#torch.set_default_device("cuda")

#Create model
# note - made some changes here to get it working on CPU..
model = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b", 
    torch_dtype=torch.float32, 
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)

def generateAnswer(question,image_path):
    #Set inputs
    system_prompt="A chat between a curious user and an assistant. The assistant gives helpful, detailed answers to the user's questions."
    text = system_prompt+' USER: <image>\n'+question+' ASSISTANT:'
    image = Image.open(image_path)

    input_ids = tokenizer(text, return_tensors='pt').input_ids
    image_tensor = model.image_preprocess(image)

    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        images=image_tensor,
        use_cache=True)[0]
    
    result=tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    return result

print(generateAnswer('Is this a cat?','images/dani.JPG'))