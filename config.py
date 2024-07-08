

datasets_config = [
    {
        "name": "wikimedia/wikipedia",
        "version": "20231101.en",
        "columns": ["text"],
        "task": "next_token_prediction"
    },
    {
        "name": "HuggingFaceFW/fineweb-edu",
        "columns": ["text"],
        "task": "next_token_prediction"
    },
    {
        "name": "alespalla/chatbot_instruction_prompts",
        "columns": ["response", "prompt"],
        "task": "response_generation"
    },
    {
        "name": "instruction-pretrain/ft-instruction-synthesizer-collection",
        "columns": ["context", "QA_list", "QA_type"],
        "task": "context_generation"
    },
    {
        "name": "vilm/Pretrain-Instruction-1",
        "columns": ["text"],
        "task": "next_token_prediction"
    },
    {
        "name": "vilm/Pretrain-Instruction-2",
        "columns": ["text"],
        "task": "next_token_prediction"
    },
    {
        "name": "xinlai/Math-Step-DPO-10K",
        "columns": ["dataset", "prompt", "initial_reason_steps", "chosen", "rejected", "full_chosen", "full_rejected", "answer"],
        "task": "reasoning_steps_prediction"
    },
    {
        "name": "Deojoandco/reward_model_anthropic",
        "columns": ["prompt", "chosen", "rejected"],
        "task": "reward_model_choice"
    },
    {
        "name": "abacusai/MetaMath_DPO_FewShot",
        "columns": ["prompt", "chosen", "rejected"],
        "task": "reward_model_choice"
    },
    {
        "name": "nthakur/multilingual-ultrafeedback-dpo-v0.1",
        "columns": ["id", "en_chosen", "en_rejected", "en_input", "source", "input", "chosen", "rejected", "language"],
        "task": "multilingual_reward_model_choice"
    },
    {
        "name": "Intel/orca_dpo_pairs",
        "columns": ["system", "question", "chosen", "rejected"],
        "task": "reward_model_choice"
    },
    {
        "name": "codeparrot/codeparrot-clean",
        "columns": ["repo_name", "path", "copies", "size", "content"],
        "task": "next_token_prediction"
    },
    {
        "name": "suvadityamuk/huggingface-transformers-code-dataset",
        "columns": ["text"],
        "task": "next_token_prediction"
    },
    {
        "name": "Fraser/python-state-changes",
        "columns": ["start", "code", "end"],
        "task": "code_generation"
    },
    {
        "name": "lucasmccabe-lmi/gpt4all_code",
        "columns": ["instruction", "input", "output"],
        "task": "instruction_to_code_generation"
    },
    {
        "name": "sahil2801/CodeAlpaca-20k",
        "columns": ["output", "instruction", "input"],
        "task": "instruction_to_code_generation"
    },
    {
        "name": "lucasmccabe-lmi/codex_math_qa_alpaca_style",
        "columns": ["instruction", "input", "output"],
        "task": "instruction_to_code_generation"
    },
    {
        "name": "lucasmccabe-lmi/instruct_to_code_alpaca_style",
        "columns": ["instruction", "input", "output"],
        "task": "instruction_to_code_generation"
    },
    {
        "name": "TokenBender/code_instructions_122k_alpaca_style",
        "columns": ["instruction", "output", "text", "input"],
        "task": "instruction_to_code_generation"
    },
    {
        "name": "HydraLM/GPTeacher_codegen_alpaca",
        "columns": ["input", "output"],
        "task": "instruction_to_code_generation"
    }
]
