%tensorflow_version 2.x
import tensorflow as tf

from transformers import (TFBertForSequenceClassification, 
                          BertTokenizer,
                          TFRobertaForSequenceClassification, 
                          RobertaTokenizer)

bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


sequence = "Systolic arrays are cool. This 🐳 is cool too."

bert_tokenized_sequence = bert_tokenizer.tokenize(sequence)
print("BERT:", bert_tokenized_sequence)


import tensorflow_datasets
data = tensorflow_datasets.load("glue/mrpc")

train_dataset = data["train"]
validation_dataset = data["validation"]


example = list(train_dataset.__iter__())[0]
print('',
    'idx:      ', example['idx'],       '\n',
    'label:    ', example['label'],     '\n',
    'sentence1:', example['sentence1'], '\n',
    'sentence2:', example['sentence2'],
)



seq0 = example['sentence1'].numpy().decode('utf-8')  # Obtain bytes from tensor and convert it to a string
seq1 = example['sentence2'].numpy().decode('utf-8')  # Obtain bytes from tensor and convert it to a string

print("First sequence:", seq0)
print("Second sequence:", seq1)


encoded_bert_sequence = bert_tokenizer.encode(seq0, seq1, add_special_tokens=True, max_length=128)

print("BERT tokenizer separator, cls token id:   ", bert_tokenizer.sep_token_id, bert_tokenizer.cls_token_id)

bert_special_tokens = [bert_tokenizer.sep_token_id, bert_tokenizer.cls_token_id]

def print_in_red(string):
    print("\033[91m" + str(string) + "\033[0m", end=' ')

print("\nBERT tokenized sequence")
output = [print_in_red(tok) if tok in bert_special_tokens else print(tok, end=' ') for tok in encoded_bert_sequence]



from transformers import glue_convert_examples_to_features

bert_train_dataset = glue_convert_examples_to_features(train_dataset, bert_tokenizer, 128, 'mrpc')
bert_train_dataset = bert_train_dataset.shuffle(100).batch(32).repeat(2)

bert_validation_dataset = glue_convert_examples_to_features(validation_dataset, bert_tokenizer, 128, 'mrpc')
bert_validation_dataset = bert_validation_dataset.batch(64)


optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


print("Fine-tuning BERT on MRPC")
bert_history = bert_model.fit(bert_train_dataset, epochs=3, validation_data=bert_validation_dataset)


print("Evaluating the BERT model")
bert_model.evaluate(bert_validation_dataset)