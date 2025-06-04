# 1B-building-engineering-llm
Fine-tuning EleutherAI/pythia-1b for building-engineering tasks using 4-bit quant + LoRA

> **A birthday gift for my dad**  
> Happy Birthday, Dad! I built this LLM assistant around building-engineering knowledge.

---

## What Is This?

- **Base model:** EleutherAI/pythia-1b-deduped  
- **Quantization:** 4-bit (BitsAndBytes) for efficiency  
- **Adapters:** LoRA to teach domain-specific concepts without retraining the entire model 
- **Training:** Code and instructions to reproduce data collection, tokenization, and fine-tuning

Over the past few months, I studied building-engineering concepts—insulation, structural design, U-values, concrete strengths, etc.—to gather high-quality data. As of today (June 4, 2025), the model is available.  Feel free to try it out.

---

## Disclaimer 

I collected data over four months and studied building engineering in depth to ensure accuracy. The full merged model is available now.

---

## Quickstart

**Clone the repo**

 If your Git setup supports Git LFS, use:

```bash

git clone https://github.com/IrfanUruchi/1B-building-engineering-llm.git
cd 1B-building-engineering-llm
```
Otherwise, download individual files from the GitHub web interface.

Use the code below to load the merged model and ask a simple question:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "merged-model/"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()

prompt = (
    "You are an experienced building engineer. When asked, answer concisely in plain English.\n\n"
    "Q: What is a structural engineer?\n"
    "A:"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
```

## License

This project (code & training recipes) is released under the Apache License 2.0.
Model weights (EleutherAI/pythia-1b-deduped) are also Apache-2.0. See [Licence](https://www.apache.org/licenses/LICENSE-2.0)


Happy Birthday, Dad!
