import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from RemoteTweetsRepository import *

remoteTweetRepo = RemoteTweetsRepository()

df = remoteTweetRepo.get_tweets_from_git_url()
batch_1 = df[:1000]
df = batch_1

print(df.head())
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer,
                                                    'distilbert-base-uncased')

# Want BERT instead of distilBERT? Uncomment the following line:
# model_class, tokenizer_class, pretrained_weights =
# (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print()

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded_tokenized_values = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
print(np.array(padded_tokenized_values).shape)

token_attention_mask = np.where(padded_tokenized_values != 0, 1, 0)
print(token_attention_mask.shape)

padded_input_ids = torch.tensor(padded_tokenized_values)
token_attention_mask = torch.tensor(token_attention_mask)

with torch.no_grad():
    last_hidden_states = model(padded_input_ids, attention_mask=token_attention_mask)



features = last_hidden_states[0][:,0,:].numpy()
features_all = last_hidden_states[0][:,:,:].numpy()
labels = df[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)



# parameters = {'C': np.linspace(0.0001, 100, 20)}
# grid_search = GridSearchCV(LogisticRegression(), parameters)
# grid_search.fit(train_features, train_labels)

# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

score = lr_clf.score(test_features, test_labels)
print(score)

from sklearn.dummy import DummyClassifier
dclf = DummyClassifier()

scores = cross_val_score(dclf, train_features, train_labels)
print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print()
