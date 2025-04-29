from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import random
import math
from tqdm import tqdm
import pandas as pd
from rc_experiment.training import _build_prompt_batch


# Helper function to mannually pad prompt for decoder only model (requires left-padding for generation)
def _build_prompt_batch(input_ids, labels, tokenizer):
    """
    • Keeps existing left pads
    • Removes everything to the right of the prompt
    • Re-pads (on the left) so the batch is rectangular again
    Returns a dict ready for model.generate().
    """
    prompt_only = []
    for seq, lab in zip(input_ids, labels):
        first_comp = (lab != -100).nonzero(as_tuple=True)[0][0].item()
        prompt_only.append(seq[:first_comp])            # ← no right pads!

    return tokenizer.pad(
        {"input_ids": prompt_only},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
        padding_side="left"   # <--- explicitly force left padding
    )



def rc_eval(test_loader, model_obj, tokenizer_obj, device, max_input_length, max_target_length):

    tokenizer_obj.padding_side = 'left'
    model_obj.eval()
    correct = 0
    total   = 0
    rows = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Load in prompt for later prediction 
            prompt_batch = _build_prompt_batch(input_ids, labels, tokenizer_obj)
            prompt_batch = {k: v.to(device) for k, v in prompt_batch.items()}

            # Generate preidciont based on prompt
            preds = model_obj.generate(
                **prompt_batch,
                max_new_tokens=max_target_length,
                pad_token_id=tokenizer_obj.pad_token_id,  # good practice
            )
                        
            # Store the test set prediction to a data frame
            for i, pred_ids in enumerate(tqdm(preds, desc="Decoding predictions", leave=False, unit="sample")):
                # Compare predictions with true completions
                # Prediction text
                pred_text = tokenizer_obj.decode(pred_ids, skip_special_tokens=True)
                
                # True text
                true_ids  = labels[i]
                true_ids  = true_ids[true_ids != -100]      # strip ignore index
                true_text = tokenizer_obj.decode(true_ids, skip_special_tokens=True)
                
                # Count the number of correct and total prediction
                if pred_text.strip() == true_text.strip():
                    correct += 1
                total += 1

                # For every item in the batch, collect prompt / pred / truth
                prompt_text = tokenizer_obj.decode(input_ids[i], skip_special_tokens=True)
                rows.append({
                    "Prompt": prompt_text.strip(),
                    "Prediction": pred_text.strip(),
                    "Ground‑Truth": true_text.strip(),
                    "Exact Match": "✅" if pred_text.strip() == true_text.strip() else "❌"
                })
    test_em_accuracy = correct / total if total else 0.0
    print(f"Test Exact Match Accuracy: {test_em_accuracy*100:.2f}% "
        f"({correct}/{total} correctly matched)")
    
    # Return the prediction data frame
    return pd.DataFrame(rows)
