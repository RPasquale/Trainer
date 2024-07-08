# To Train a Model

cd .Trainer/

To train on all datasets with multiple loops using the custom model:
python train.py --num_loops 2 --model_type custom

To train on a specific dataset using the custom model:
python train.py --dataset_name "alespalla/chatbot_instruction_prompts" --num_loops 2 --model_type custom
