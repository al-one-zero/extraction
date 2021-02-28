import pandas as pd
import numpy as np
import re
import fasttext
from emot.emo_unicode import UNICODE_EMO

#hashtags regex "#..."
hashtag_re = r"#([\w\+_-]*)"
#mentions regex "@..."
mention_re = r"@(\w*)"
#http/https links regex
link_re = r"(https?://|(https?://)?bit.ly)[^\s]+"

#remove "\n" from the ending of a tweet
def remove_slash_n(tweet):
    return tweet[:-1] if(tweet.endswith("\n")) else tweet

# replace http/https links with a unique word
def replace_links(tweet, rep="_LINK_"):
    return re.sub(link_re, rep, tweet)

#remove the #/@ prefixes of hashtags/mentions
def remove_hashtags_and_mentions(tweet):
    s = re.sub(hashtag_re, r"\1", tweet)
    return re.sub(mention_re, r"\1", s)

#convert emojis in a text to some meaningful expressions
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "(" + (UNICODE_EMO[emot].replace(",","").replace(":","")\
                                                .replace("_"," ")) + ")")
    return text

#perform all the previous operations on a given tweets
def preprocess_tweet(tweet):
    return convert_emojis(replace_links(remove_slash_n(remove_hashtags_and_mentions(tweet))))\
            if tweet != None else None

#find list of hashtags
def find_hashtags(tweet):
    return [h for h in re.findall(hashtag_re, tweet) if h != ""]

#find list of mentions
def find_mentions(tweet):
    return [h for h in re.findall(mention_re, tweet) if h != ""]

#convert the whole tweet to lowercase, except for the company names
#this improves language prediction
def tweet_to_lower(tw):
    return tw.lower().replace("microsoft", "Microsoft").replace("google", "Google")\
        .replace("apple", "Apple").replace("twitter", "Twitter")

#given a dataframe that contains at least a "Tweet" column, add the following columns :
#Tweet : the preprocessing version of the original column
#Hashtags : the list of hashtags
#Mentions : the list of mentions
#Language : the language in which the tweet is written
#LanguageProbabiblity : the probabiblity that the estimated language is correct
def preprocess_dataset(df_, fasttext_model_location='../data/lid.176.bin'):
    df = df_.copy()
    df["Hashtags"] = df.Tweet.apply(find_hashtags)
    df["Mentions"] = df.Tweet.apply(find_mentions)
    df["Tweet"] = df.Tweet.apply(preprocess_tweet)
    
    model = fasttext.load_model(fasttext_model_location)
    
    df["Language"] = df.Tweet.apply(lambda x :
                                    model.predict(tweet_to_lower(x), k=1)[0][0][len("__label__"):])
    df["LanguageProbability"] = df.Tweet.apply(lambda x :
                                    model.predict(tweet_to_lower(x), k=1)[1][0])
    return df