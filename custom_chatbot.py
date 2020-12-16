from transformers import AutoModelWithLMHead, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")

'''
sentence_list = [
    "I heard you won the cricket match.",
    "I did!",
    "Awesome. Who did you play against?",
    "I played against the Aussies.",
    "Wow ! Was it a tough game?",
    "It was a tough game. It went on till the last over. They almost won.",
    "Where was the match?",
    "It was in Canberra",
    "What is the format of the game?"
]
'''

sentence_list = [
                 "What is your name?",
                 "I'm Jacob. What is yours?",
                 "Hi Jacob! my name is Charlie. Are you a survivor?",
                 "Hi Charlie! Yes, I am a survivor of a plane crash.",
                 "How did you get on this island?",
                 "I was washed off to the shore with the wreckage.",
                 "Do you know what place is this?",
                 "Yes, we are in Ireland.",
                 "Where were you travelling to?",
                 "I was going to Los Angeles",
                 "Do you have your family there?",
                 "No. Just for some business.",
                 "Are you a savage?",
                 "No, I am not.",
                 "Are we in Ireland?",
                 "Yes, we're in Ireland.",
                 "Where in Ireland?"

]

all_sentences_string =""
for sentence in sentence_list:
    all_sentences_string = all_sentences_string+sentence+tokenizer.eos_token
    
print ("All sentences concatenated with EOS token:\n")
print (all_sentences_string)

tokenized_all_sentences_string = tokenizer.encode(all_sentences_string, return_tensors='pt')
reply_predicted =  model.generate(tokenized_all_sentences_string, max_length=1000)
prefix_length =  tokenized_all_sentences_string.shape[-1]

decoded_reply_predicted_with_input = tokenizer.decode(reply_predicted[0], skip_special_tokens=True)
decoded_reply_predicted = tokenizer.decode(reply_predicted[:,prefix_length:][0], skip_special_tokens=True)


print ("\n\nPredicted reply along with initial input: ")
print (decoded_reply_predicted_with_input)

print ("\n\nPredicted reply: ")
print (decoded_reply_predicted)
