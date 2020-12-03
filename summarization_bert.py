
def load():
	git clone https://github.com/chriskhanhtran/bert-extractive-summarization.git
	%cd bert-extractive-summarization
	!pip install -r requirements.txt

	!wget -O "checkpoints/bertbase_ext.pt" "https://www.googleapis.com/drive/v3/files/1t27zkFMUnuqRcsqf2fh8F1RwaqFoMw5e?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
	!wget -O "checkpoints/distilbert_ext.pt" "https://www.googleapis.com/drive/v3/files/1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
	!wget -O "checkpoints/mobilebert_ext.pt" "https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"


	from newspaper import Article
	import torch
	from models.model_builder import ExtSummarizer
	from ext_sum import summarize
	import textwrap
	import nltk
	nltk.download('punkt')



url = "https://www.cnn.com/2020/05/29/tech/facebook-violence-trump/index.html" #@param {type: 'string'}
article = Article(url)
article.download()
article.parse()
print(wrapper.fill(article.text))
t = article.text

#@param ['bertbase', 'distilbert', 'mobilebert']

def summarize(text, model = 'distilbert'):

	with open('raw_data/input.txt', 'w') as f:
		f.write(text)
	# Load model
	model_type = model 
	checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
	model = ExtSummarizer(checkpoint=checkpoint, bert_type=model_type, device="cpu")

	%%time
	# Run summarization
	input_fp = 'raw_data/input.txt'
	result_fp = 'results/summary.txt'
	summary = summarize(input_fp, result_fp, model, max_length=3)

	# Print summary
	wrapper = textwrap.TextWrapper(width=80)
	print(wrapper.fill(summary))