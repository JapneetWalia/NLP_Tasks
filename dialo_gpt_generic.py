
def chatbot():

    import torch
    from transformers import AutoModelWithLMHead, AutoTokenizer
	

	tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
	
	model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
	
	for step in range(2):
		new_user_inputs_ids = tokenizer.encode(input(">> User: ") + tokenizer.eos_token, return_tensors='pt')
		
		bot_input_ids = torch.cat([chat_history_ids, new_user_inputs_ids], dim=1) if step > 0 else new_user_inputs_ids
		
		chat_history_ids = model.generate(bot_input_ids, max_length=2000, pad_token_id=tokenizer.eos_token_id)
		
		print("DialogGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
