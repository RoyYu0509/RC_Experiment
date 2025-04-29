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
        { "input_ids": prompt_only },
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


# Training Loop
def casual_llm_train(model_name,  # For storing results
                     model_obj, tokenizer_obj, optimizer_obj,  # training objects
                     train_loader, val_loader,  # data loader
                     device,  
                     max_target_length,  # maximum output length (in tokens)
                     num_epochs, patience=2, min_delta = 0.0  # Some training config
                     ):
    """
    Description: 
        Train the `model_obj` with `optimizer_obj` in a pytorch 
        customized training loop, using data from `train_loader` and `val_loader` 
        on `device`. 
    
        Early stopping is implemented in validation step with `patience` and `min_delta`

    Parameter:
        - Omitted....
    """

    train_losses = []
    val_losses = []
    val_accuracies = []  # track exact-match accuracy on validation

    best_val_loss = float('inf')
    patience_counter = 0


    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # --- Training ---
        model_obj.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model_obj(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer_obj.zero_grad()
            loss.backward()
            optimizer_obj.step()

            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model_obj.eval()
        val_loss_total = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Load in prompt for later prediction 
                prompt_batch = _build_prompt_batch(input_ids, labels, tokenizer_obj)
                prompt_batch = {k: v.to(device) for k, v in prompt_batch.items()}

                preds = model_obj.generate(
                    **prompt_batch,
                    max_new_tokens=max_target_length,
                    pad_token_id=tokenizer_obj.pad_token_id,  # good practice
                )

                # Compute validation loss
                outputs = model_obj(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss_total += outputs.loss.item()
                
                for i, pred_ids in enumerate(preds):
                    pred_ids = pred_ids.tolist()
                    # Remove the prompt part from the generated sequence
                    prompt_len = (labels[i] != -100).nonzero(as_tuple=True)[0][0].item()
                    generated_tokens = pred_ids[prompt_len:]
                    pred_text = tokenizer_obj.decode(generated_tokens, skip_special_tokens=True)

                    # Decode the true completion (labels where label != -100)
                    true_ids = labels[i][labels[i] != -100]
                    true_text = tokenizer_obj.decode(true_ids.tolist(), skip_special_tokens=True)

                    # .contain
                    if true_text.strip() in pred_text.strip():
                        correct += 1
                    total += 1

        avg_val_loss = val_loss_total / len(val_loader)
        val_em = correct / total if total > 0 else 0.0  # exact match accuracy
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_em)

        print(f"Epoch {epoch+1:02}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val EM: {val_em*100:.2f}%")

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            saving_dir = f"./best_model/{model_name}"
            model_obj.save_pretrained(saving_dir)  # Save immediately
            tokenizer_obj.save_pretrained(saving_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Return where to load the best model condif
    return saving_dir, train_losses, val_losses, val_accuracies

import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, title='Training and Validation Loss'):
    """
    Plot training and validation losses over epochs with a custom title.

    Parameters:
        train_losses (list of float): Training loss values per epoch.
        val_losses (list of float): Validation loss values per epoch.
        title (str): Title for the plot.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()