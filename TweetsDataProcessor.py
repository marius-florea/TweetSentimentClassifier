import pandas as pd
import numpy as np
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

class TweetsDataProcessor:

    def tokenize_tweets(self, tweets_df: pd.DataFrame) -> (torch.Tensor, torch.Tensor):
        tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer,
                                                            'distilbert-base-uncased')

        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        tokenized = tweets_df.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

        max_len = len(max(tokenized.values, key=len))
        padded_tokenized_values = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
        print(np.array(padded_tokenized_values).shape)

        token_attention_mask = np.where(padded_tokenized_values != 0, 1, 0)
        print(token_attention_mask.shape)

        padded_input_ids = torch.tensor(padded_tokenized_values)
        token_attention_mask = torch.tensor(token_attention_mask)

        return padded_input_ids, token_attention_mask

