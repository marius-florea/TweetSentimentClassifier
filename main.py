from TweetsManager import TweetsManager

tweets_manager = TweetsManager()
# tweets_manager.load_data_and_train_model()
# hashtag = "#bitcoin lang:en -is:retweet"
# query = "#covid19 lang:en -is:retweet"
# tweets_manager.get_tweets_by_hashtag(hashtag)
# tweets_manager.load_tweets_from_json('bitcoin-tweets.json')
# tweets_manager.load_data_load_distilbert_model_make_prediction()
# tweets_manager.load_data_predict_using_sent_pipeline()
# tweets_manager.load_data_fine_tune_model_for_seq_classification()
tweets_manager.predict_from_saved_model()