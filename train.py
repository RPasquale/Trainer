import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from config import datasets_config
from data_loader import load_dataset_by_config, get_data_loader
from tokenizer import tokenizer, data_collator
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import argparse
from gpt import GPT, GPTConfig

def plot_losses(loss_dict, output_dir, dataset_name):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_dict['train_loss'], label='Train Loss')
    plt.plot(loss_dict['eval_loss'], label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Train and Validation Loss for {dataset_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_loss.png"))
    plt.close()

def main(num_loops=1, dataset_name=None, model_type="gpt2"):
    # Load your custom model
    if model_type == "custom":
        model = GPT(GPTConfig())  # Replace this with your model's configuration if necessary
    else:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    output_dir = "./results"
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
    )

    if dataset_name:
        datasets_to_train = [config for config in datasets_config if config['name'] == dataset_name]
    else:
        datasets_to_train = datasets_config

    if not datasets_to_train:
        raise ValueError(f"No matching dataset found for name: {dataset_name}")

    for loop in range(num_loops):
        for config in datasets_to_train:
            print(f"Training on dataset: {config['name']} (Loop {loop+1})")
            dataset_config_name = config['name'].replace('/', '_')

            dataset, columns, task = load_dataset_by_config(config)
            processed_dataset = get_data_loader(dataset, columns, task)
            
            # Remove only columns that exist in the dataset
            columns_to_remove = [col for col in processed_dataset.column_names if col in ["input_text", "target_text"]]
            
            def tokenize_and_align_labels(examples):
                tokenized_inputs = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=512)
                tokenized_inputs["labels"] = tokenizer(examples["target_text"], truncation=True, padding="max_length", max_length=512)["input_ids"]
                return tokenized_inputs
            
            tokenized_dataset = processed_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=columns_to_remove)

            data_loader = DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size)

            loss_dict = defaultdict(list)
            
            class CustomTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    logits = outputs.get("logits")
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    return (loss, outputs) if return_outputs else loss

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if 'loss' in logs:
                        loss_dict['train_loss'].append(logs['loss'])
                    if 'eval_loss' in logs:
                        loss_dict['eval_loss'].append(logs['eval_loss'])
                    super().on_log(args, state, control, logs, **kwargs)

            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            trainer.train()
            trainer.evaluate()

            plot_losses(loss_dict, output_dir, dataset_config_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 on specified datasets.")
    parser.add_argument('--num_loops', type=int, default=1, help='Number of loops of epochs to run.')
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset to train on. If not specified, train on all datasets.')
    parser.add_argument('--model_type', type=str, default="gpt2", choices=["gpt2", "custom"], help='Type of model to use for training.')

    args = parser.parse_args()
    main(num_loops=args.num_loops, dataset_name=args.dataset_name, model_type=args.model_type)

# To train on all datasets with multiple loops using the custom model:
# python train.py --num_loops 2 --model_type custom

# To train on a specific dataset using the custom model:
# python train.py --dataset_name "alespalla/chatbot_instruction_prompts" --num_loops 2 --model_type custom

