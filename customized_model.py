from transformers import pipeline, Conversation
import torch

compute_dtype = getattr(torch, "float16")

def player_initialization(model_id, prompt_role_init):
    role = pipeline(
        "conversational", 
        model=model_id,
        model_kwargs={
            # "torch_dtype": torch.bfloat16,
            "quantization_config": {"load_in_4bit": True,
            						"bnb_4bit_quant_type": "nf4",
            						"bnb_4bit_compute_dtype": compute_dtype,
            						"bnb_4bit_use_double_quant": True,
                                    # "load_in_8bit_fp32_cpu_offload": True,
                                    "llm_int8_enable_fp32_cpu_offload": True
                                    },
            "low_cpu_mem_usage": True,
        },
        # device="cuda",
        device_map="auto",
        )

    history_conversation = Conversation(prompt_role_init)
    history_conversation = role(
        history_conversation,
        max_new_tokens=256,
        pad_token_id=role.tokenizer.eos_token_id)

    return role, history_conversation


def judge_initialization(model_id, prompt_role_init, prompt_puzzle):
	role = pipeline(
		"conversational", 
		model=model_id,
	    model_kwargs={
	    	# "torch_dtype": torch.bfloat16,
			"quantization_config": {"load_in_4bit": True,
            						"bnb_4bit_quant_type": "nf4",
            						"bnb_4bit_compute_dtype": compute_dtype,
            						"bnb_4bit_use_double_quant": True,
									# "load_in_8bit_fp32_cpu_offload": True,
									"llm_int8_enable_fp32_cpu_offload": True
									},
			"low_cpu_mem_usage": True,
	    },
	    # device="cuda",
	    device_map="auto",
	    )

	history_conversation = Conversation(prompt_role_init)
	history_conversation.add_message({"role": "user", "content": prompt_puzzle})

	history_conversation = role(
		history_conversation,
		max_new_tokens=256,
		pad_token_id=role.tokenizer.eos_token_id)

	return role, history_conversation
