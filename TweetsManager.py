import tweepy
import pandas as pd
from RemoteTweetsRepository import *
from TweetsDataProcessor import TweetsDataProcessor
import torch
import transformers as ppb  # pytorch transformers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import pipeline
from datasets import Dataset
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_metric
import numpy as np

# your bearer token #TODO remove on commit
MY_BEARER_TOKEN = ""

class TweetsManager:
    remote_tweets_repo = RemoteTweetsRepository()
    client = tweepy.Client(bearer_token=MY_BEARER_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __init__(self):
        print("__init__")

    #Getters
    def get_tweets_from_git_url(self) -> pd.DataFrame:
        return TweetsManager.remote_tweets_repo.get_tweets_from_git_url()

    def load_tweets(self):
        tweets_df = self.get_tweets_from_git_url()
        tweets_df.rename(columns={0: 'text', 1: 'label'}, inplace=True)
        subsample_df = tweets_df[:8] #TODO comment this or set a higher value
        tweets_df = subsample_df
        return tweets_df

    def load_and_pad_unlabeled_tweets(self):
        tweets_df = self.load_tweets_from_json('bitcoin-tweets_short.json')
        # subsample_df = tweets_df[:1000]
        # tweets_df = subsample_df
        tweets_df = tweets_df['content']
        padded_input_ids, token_attention_mask = self.tokenize_tweets(tweets_df)
        return padded_input_ids, token_attention_mask

    def load_tweets_from_json(self, json):
        tweets_df = pd.read_json(json, lines=True)
        return tweets_df

    def get_tweets_by_hashtag(self, query: str) -> pd.DataFrame:
        # authenticatione

        search_query = query
        start_time = "2022-05-24T13:00:00Z"
        end_time = "2022-05-31T00:00:00Z"
        tweets = TweetsManager.client.search_recent_tweets(query=query,
                                                           start_time=start_time,
                                                           end_time=end_time,
                                                           tweet_fields=["created_at", "text", "source"],
                                                           user_fields=["name", "username", "location", "verified",
                                                                        "description"],
                                                           max_results=10000,
                                                           expansions='author_id'
                                                           )

        # create a list of records
        tweet_info_ls = []
        # iterate over each tweet and corresponding user details
        for tweet, user in zip(tweets.data, tweets.includes['users']):
            tweet_info = {
                'created_at': tweet.created_at,
                'text': tweet.text,
                'source': tweet.source,
                'name': user.name,
                'username': user.username,
                'location': user.location,
                'verified': user.verified,
                'description': user.description
            }
            tweet_info_ls.append(tweet_info)
        # create dataframe from the extracted records
        tweets_df = pd.DataFrame(tweet_info_ls)
        # display the dataframe
        tweets_df.head()

        return tweets_df

    #Processeing
    def tokenize_tweets(self, tweets_df: pd.DataFrame) -> (torch.Tensor, torch.Tensor):
        # pass tweets to tweets data processor
        tweetsProcessor = TweetsDataProcessor()
        padded_input_ids, token_attention_mask = tweetsProcessor.tokenize_tweets(tweets_df)
        return padded_input_ids, token_attention_mask

    def concat_array_to_df(self, tweets_df, arr):
        df_from_array = pd.DataFrame(arr, columns=['prediction'])
        concatenated_df = pd.concat([tweets_df, df_from_array], axis=1)
        matching_sentiments = concatenated_df['prediction'] == (concatenated_df['label'])
        nr_of_matching_sentiments = matching_sentiments.value_counts()[True]
        total_sentiments = len(matching_sentiments)
        matching_percentage = nr_of_matching_sentiments / total_sentiments
        print('matching percentage ', matching_percentage)
        return concatenated_df
    # Tokenize and encode the dataset
    def tokenize(self, batch):
        tokenized_batch = TweetsManager.tokenizer(batch['text'], padding=True, truncation=True, max_length=128)
        return tokenized_batch

    #utility
    def get_available_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    #model training predictions ...
    def load_data_predict_using_sent_pipeline(self):
        sentiment_pipeline = pipeline("sentiment-analysis")
        tweets_df = self.load_tweets()
        tweets_list = tweets_df[0].tolist()
        result = sentiment_pipeline(tweets_list)
        print()
        dfr = pd.DataFrame(result)
        classification_dict = {'POSITIVE': 1, 'NEGATIVE': 0}
        horizontal_concat = pd.concat([tweets_df, dfr], axis=1)
        horizontal_concat['label'] = horizontal_concat['label'].apply(lambda x: classification_dict[x])
        matching_sentiments = horizontal_concat[1] == (horizontal_concat['label'])
        nr_of_matching_sentiments = matching_sentiments.value_counts()[True]
        total_sentiments = len(matching_sentiments)
        matching_percentage = nr_of_matching_sentiments / total_sentiments
        print('matching percentage ', matching_percentage)
        model = sentiment_pipeline.model

        return

    def distilbert_make_prediction(self, model, padded_input_ids, token_attention_mask, labels):
        # encoding of the data
        with torch.no_grad():
            last_hidden_states = model(padded_input_ids, attention_mask=token_attention_mask)

        features = last_hidden_states[0][:, 0, :].numpy()

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
        # fitting logistic regression on the encoded data
        lr_clf = LogisticRegression()
        lr_clf.fit(train_features, train_labels)
        # see how well it fits
        score = lr_clf.score(test_features, test_labels)
        print(score)

    def load_data_load_distilbert_model_make_prediction(self):
        tweets_df = self.load_tweets()
        padded_input_ids, token_attention_mask = self.tokenize_tweets(tweets_df['text'])
        model_class, pretrained_weights = (ppb.DistilBertModel,
                                           'distilbert-base-uncased')
        ## Want BERT instead of distilBERT? Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights =
        # (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        model = model_class.from_pretrained(pretrained_weights)
        # Check the output
        # print(dataset_enc["train"].column_names)
        labels = tweets_df['label']
        self.distilbert_make_prediction(model, padded_input_ids, token_attention_mask, labels)

        # #loading some unlabeled data
        # padded_input_ids_unlbl, token_attention_mask_unlbl = self.load_and_pad_unlabeled_tweets()
        #
        # #encoding the data
        # with torch.no_grad():
        #     last_hidden_states_unlbl = model(padded_input_ids_unlbl, attention_mask=token_attention_mask_unlbl)
        # #extracting the class encoding
        # features_unlbl = last_hidden_states_unlbl[0][:, 0, :].numpy()
        # #prediction
        # tweets_sentiment_predictions = lr_clf.predict(features_unlbl)
        #
        # #concatenate the predictions and the tweets into a dataframe
        # df_to_pred = self.load_tweets_from_json('bitcoin-tweets_short.json')
        # prediction_df = pd.DataFrame({'tweet': df_to_pred['content'].to_numpy(), 'label': tweets_sentiment_predictions})
        # print()

    def evaluate_auto_model_for_seq_classification(self, model, eval_dataloader):
        metric = load_metric("glue", "mrpc")
        # iteratively evaluate the model
        model.eval()
        device = self.get_available_device()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        print(metric.compute())

    def predict_from_saved_model(self):
        model = AutoModelForSequenceClassification.from_pretrained("distilbert_sentiments_model_3000rows")
        tweets_df = self.load_tweets()
        # padded_input_ids, token_attention_mask = self.tokenize_tweets(tweets_df['text'])
        labels = tweets_df['label']
        # the shape changed of the output of the model cause of my training
        dataset = Dataset.from_pandas(tweets_df).train_test_split(train_size=0.8, seed=43)
        # Make a list of columns to remove before tokenization
        cols_to_remove = [col for col in dataset["train"].column_names if col != "label"]

        dataset = dataset.class_encode_column("label")
        dataset_enc = dataset.map(self.tokenize, batched=True, remove_columns=cols_to_remove,
                                  num_proc=4)
        dataset_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        data_collator = DataCollatorWithPadding(tokenizer=TweetsManager.tokenizer)

        eval_dataloader = DataLoader(dataset_enc["test"], batch_size=8, collate_fn=data_collator)

        self.evaluate_auto_model_for_seq_classification(model, eval_dataloader)

        device = self.get_available_device()
        tweets_text = tweets_df['text'].to_list()
        inputs = self.tokenizer(tweets_text, padding=True, truncation=True, return_tensors="pt").to(
            device)  # Move the tensor to the GPU
        # make prediction
        outputs = model(**inputs)
        print(outputs)
        # conv logits to class probabilities
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions_np = predictions.argmax(dim=1).detach().numpy()
        df_with_predictions = self.concat_array_to_df(tweets_df, predictions_np)

    def load_data_fine_tune_model_for_seq_classification(self):
        tweets_df = self.load_tweets()

        dataset = Dataset.from_pandas(tweets_df).train_test_split(train_size=0.8, seed=43)
        # Make a list of columns to remove before tokenization
        cols_to_remove = [col for col in dataset["train"].column_names if col != "label"]
        print(cols_to_remove)

        dataset = dataset.class_encode_column("label")
        dataset_enc = dataset.map(self.tokenize, batched=True, remove_columns=cols_to_remove,
                                  num_proc=4)
        dataset_enc.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        print(dataset_enc["train"].column_names)

        data_collator = DataCollatorWithPadding(tokenizer=TweetsManager.tokenizer)
        train_dataloader = DataLoader(dataset_enc["train"], shuffle=True, batch_size=8,
                                      collate_fn=data_collator)
        eval_dataloader = DataLoader(dataset_enc["test"], batch_size=8, collate_fn=data_collator)
        num_labels = dataset["train"].features["label"].num_classes
        print(f"Number of labels:{num_labels}")
        # load model from checkpoint
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                   num_labels=num_labels)
        # Model parameters
        learning_rate = 5e-5
        num_epochs = 5
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        # learning rate shceduler
        num_training_batches = len(train_dataloader)
        num_training_steps = num_epochs * num_training_batches
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        device = self.get_available_device()
        print("runs on", device)
        # move model to device
        model.to(device)

        progress_bar = tqdm(range(num_training_steps))
        # train the model with pytorch training loop
        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch_dict = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(**batch_dict)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                progress_bar.update()
                progress_bar.update(1)
            lr_scheduler.step()

        #prediction code
        # tweets_text = tweets_df['text'].to_list()
        # inputs = self.tokenizer(tweets_text, padding=True, truncation=True, return_tensors="pt").to(
        #     device)  # Move the tensor to the GPU
        # # make prediction
        # outputs = model(**inputs)
        # print(outputs)
        # # conv logits to class probabilities
        # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # predicted_classes = torch.argmax(predictions, dim=1)
        # print(predictions)

        # Save model to disk
        model.save_pretrained("distilbert_sentiments_model")
