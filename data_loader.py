
from datasets import load_dataset

def load_dataset_by_config(config):
    if "version" in config:
        if "config_name" in config:
            dataset = load_dataset(config["name"], config["config_name"], config["version"], trust_remote_code=True)
        else:
            dataset = load_dataset(config["name"], config["version"], trust_remote_code=True)
    else:
        if "config_name" in config:
            dataset = load_dataset(config["name"], config["config_name"], trust_remote_code=True)
        else:
            dataset = load_dataset(config["name"], trust_remote_code=True)

    # We will always use the 'train' split for training
    dataset = dataset["train"]

    columns = config["columns"]
    task = config["task"]

    return dataset, columns, task

def get_data_loader(dataset, columns, task):
    # Define specific processing based on task
    process_function = None

    if task == "next_token_prediction":
        def process_function(examples):
            return {"input_text": examples[columns[0]], "target_text": examples[columns[0]]}
    elif task == "response_generation":
        def process_function(examples):
            return {"input_text": examples[columns[1]], "target_text": examples[columns[0]]}
    elif task == "context_generation":
        def process_function(examples):
            input_texts = []
            target_texts = []
            for i in range(len(examples[columns[0]])):
                context = examples[columns[0]][i]
                qa_list = examples[columns[1]][i]

                if isinstance(context, list):
                    context = " ".join([str(item) for item in context])
                if isinstance(qa_list, list):
                    qa_list = " ".join([str(item) for item in qa_list])

                input_texts.append(qa_list)
                target_texts.append(context)
            
            return {"input_text": input_texts, "target_text": target_texts}
    elif task == "reasoning_steps_prediction":
        def process_function(examples):
            input_texts = []
            target_texts = []
            for i in range(len(examples["prompt"])):
                input_text = examples["prompt"][i] + " " + examples["initial_reason_steps"][i]
                target_text = examples["full_chosen"][i] if examples["full_chosen"][i] else examples["full_rejected"][i]
                input_texts.append(input_text)
                target_texts.append(target_text)
            return {"input_text": input_texts, "target_text": target_texts}
    elif task == "reward_model_choice":
        def process_function(examples):
            input_texts = []
            target_texts = []
            for i in range(len(examples["prompt"])):
                input_text = examples["prompt"][i]
                target_text = examples["chosen"][i] if examples["chosen"][i] else examples["rejected"][i]
                input_texts.append(input_text)
                target_texts.append(target_text)
            return {"input_text": input_texts, "target_text": target_texts}
        
    elif task == "reward_model_choice_orca":
        def process_function(examples):
            input_texts = []
            target_texts = []
            for i in range(len(examples[columns[1]])):
                input_text = examples[columns[0]][i] + " " + examples[columns[1]][i]
                target_text = examples["chosen"][i] if examples["chosen"][i] else examples["rejected"][i]
                input_texts.append(input_text)
                target_texts.append(target_text)
            return {"input_text": input_texts, "target_text": target_texts}
        
    elif task == "multilingual_reward_model_choice":
        def process_function(examples):
            input_texts = []
            target_texts = []
            for i in range(len(examples["input"])):
                input_text = examples["input"][i]
                target_text = examples["chosen"][i] if examples["chosen"][i] else examples["rejected"][i]
                input_texts.append(input_text)
                target_texts.append(target_text)
            return {"input_text": input_texts, "target_text": target_texts}
    elif task == "code_generation":
        def process_function(examples):
            input_texts = []
            target_texts = []
            for i in range(len(examples["start"])):
                input_text = examples["start"][i]
                target_text = examples["code"][i] + " " + examples["end"][i]
                input_texts.append(input_text)
                target_texts.append(target_text)
            return {"input_text": input_texts, "target_text": target_texts}
    elif task == "instruction_to_code_generation":
        def process_function(examples):
            input_texts = []
            target_texts = []
            for i in range(len(examples["instruction"])):
                input_text = examples["instruction"][i] + " " + examples["input"][i]
                target_text = examples["output"][i]
                input_texts.append(input_text)
                target_texts.append(target_text)
            return {"input_text": input_texts, "target_text": target_texts}
        



    if process_function is None:
        raise ValueError(f"Task '{task}' is not recognized. Please ensure the task is correctly specified in the configuration.")

    processed_dataset = dataset.map(process_function, batched=True)
    return processed_dataset
