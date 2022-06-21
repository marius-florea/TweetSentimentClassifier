
tweetsGitUrl = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv'
import pandas as pd

class RemoteTweetsRepository:

    def get_tweets_from_git_url(self) -> pd.DataFrame :
        tweets_df = pd.read_csv(tweetsGitUrl, delimiter='\t', header=None)
        return tweets_df


