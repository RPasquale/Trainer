
from datasets import load_dataset

def load_dataset_by_config(config):
    if "version" in config:
        dataset = load_dataset(config["name"], config["version"])
    else:
        dataset = load_dataset(config["name"])

    # We will always use the 'train' split for training
    dataset = dataset["train"]

    columns = config["columns"]
    task = config["task"]

    return dataset, columns, task

def get_data_loader(dataset, columns, task):
    # Define specific processing based on task
    if task == "next_token_prediction":
        def process_function(examples):
            return {"input_text": examples[columns[0]], "target_text": examples[columns[0]]}
    elif task == "response_generation":
        def process_function(examples):
            return {"input_text": examples[columns[1]], "target_text": examples[columns[0]]}
    elif task == "context_generation":
        def process_function(examples):
            return {"input_text": examples[columns[1]], "target_text": examples[columns[0]]}
    elif task == "reasoning_steps_prediction":
        def process_function(examples):
            return {
                "input_text": examples["prompt"] + " " + examples["initial_reason_steps"],
                "target_text": examples["full_chosen"] if examples["full_chosen"] else examples["full_rejected"]
            }
    elif task == "reward_model_choice":
        def process_function(examples):
            return {
                "input_text": examples["prompt"],
                "target_text": examples["chosen"] if examples["chosen"] else examples["rejected"]
            }
    elif task == "multilingual_reward_model_choice":
        def process_function(examples):
            return {
                "input_text": examples["input"],
                "target_text": examples["chosen"] if examples["chosen"] else examples["rejected"]
            }
    elif task == "code_generation":
        def process_function(examples):
            return {"input_text": examples["start"], "target_text": examples["code"] + " " + examples["end"]}
    elif task == "instruction_to_code_generation":
        def process_function(examples):
            return {
                "input_text": examples["instruction"] + " " + examples["input"],
                "target_text": examples["output"]
            }

    processed_dataset = dataset.map(process_function, batched=True)
    return processed_dataset
