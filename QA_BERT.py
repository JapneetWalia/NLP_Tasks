# from QA_BERT import qa_bert, test_qa

'''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
'''




def qa_bert(question, answer_text):


	import torch
	from transformers import BertForQuestionAnswering
	from transformers import BertTokenizer


	model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
	tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
	
	input_ids = tokenizer.encode(question, answer_text)
	print('Query has {:,} tokens.\n'.format(len(input_ids))) # Report how long the input sequence is
	sep_index = input_ids.index(tokenizer.sep_token_id) # ======== Set Segment IDs ======== # Search the input_ids for the first instance of the `[SEP]` token.
	num_seg_a = sep_index + 1 # The number of segment A tokens includes the [SEP] token istelf.
	num_seg_b = len(input_ids) - num_seg_a # The remainder are segment B.


	segment_ids = [0]*num_seg_a + [1]*num_seg_b # Construct the list of 0s and 1s.

	assert len(segment_ids) == len(input_ids) # There should be a segment_id for every input token.
	
	out = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
	start_scores = out['start_logits']
	end_scores = out['end_logits']


	answer_start = torch.argmax(start_scores) # ======== Reconstruct Answer ======== # Find the tokens with the highest `start` and `end` scores.
	answer_end = torch.argmax(end_scores) 
	tokens = tokenizer.convert_ids_to_tokens(input_ids) # Get the string versions of the input tokens.


	answer = tokens[answer_start] # Start with the first token.
	for i in range(answer_start + 1, answer_end + 1):
		# If it's a subword token, then recombine it with the previous token.
		if tokens[i][0:2] == '##':
			answer += tokens[i][2:]

		# Otherwise, add a space then the token.
			
		else:
			answer += ' ' + tokens[i] 
	print('Answer: "' + answer + '"')




def test_qa():
	text = 'Gemini Solutions is an IT Consulting and Product Development firm. Our services provide clients\
 with a flexibility to choose from an array of automation and application development solutions as well\
  as giving them an option to choose from outsourcing, onshore or offshore engagement models. Gemini offers\
   several management services and is able to combine our range of services to suit a diverse range\
    of needs. We cater to the diversified portfolio of clients across sectors such as banking & financial\
     services, retail, healthcare, education and government sector. We are proud to say that we have a\
      well-structured IT community that has been handpicked from the best colleges across India who keep\
       abreast with today’s rapidly changing and ever-evolving technological advancements. We strive to\
        continuously provide customizable, affordable and quality products & services to our patrons through\
         our creative & skilled teams who demonstrate an inherent agility towards projects. CMT \
         (Comprehensive Monitoring Tool) is a tool meant to ensure that your IT operations keep running\
          smoothly and without hitches. It’s a monitoring tool that allows you to monitor the entire\
           production environment and infrastructure very closely and generates notifications as soon as\
            any issues are identified either with the infrastructure, the models that are running or the\
             data itself. What differentiates this tool from the run-of-the-mill tools is how it embeds\
              machine learning thus being able to predict a failure even before it occurs.'


	ans = 'y'
	while ans == 'y':
  		print('User:')
  		question = input()
  		qa_bert(question, text) 
  		print('\n\nAnymore questions? (y/n)')
  		ans = input()



'''
# Questions
What are the quality standards of gemini?
What are the products of gemini?
What is CRT? what's its use?          

'''