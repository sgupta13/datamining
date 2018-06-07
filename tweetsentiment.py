import sys,tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt

consumerKey = '***'
consumerSecret = '***'
accessToken = '***'
accessTokenSecret = '***'
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

# input the term to be searched and how many tweets to search
searchTerm = input("Enter Tag to search about: ")
noOfTerms = int(input("How many tweets to search: "))

tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(noOfTerms)
polarity = 0
positive = 0
negative = 0
neutral = 0

def percentage(numberofitems, allitems):
    return 100 * (numberofitems/allitems)

# iterating through tweets fetched
for tweet in tweets:
    analysis = TextBlob(tweet.text)
    polarity += analysis.sentiment.polarity
    if (analysis.sentiment.polarity == 0):
        neutral += 1
    elif (analysis.sentiment.polarity > 0):
        positive += 1
    elif (analysis.sentiment.polarity < 0):
        negative += 1

# finding average of how people are reacting
positivepercent = percentage(positive, noOfTerms)
negativepercent = percentage(negative, noOfTerms)
neutralpercent = percentage(neutral, noOfTerms)

if (polarity == 0):
    print("Neutral")
elif (polarity > 0):
    print("Positive")
elif (polarity < 0):
    print("Negative")

labels = ['Positive [' + str(positivepercent) + '%]', 'Neutral [' + str(neutralpercent) + '%]', 'Negative [' + str(negativepercent) + '%]']
sizes = [positive, neutral, negative]
colors = ['lightgreen','yellow','red']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches, labels, loc="best")
plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(noOfTerms) + ' Tweets.')
plt.axis('equal')
plt.tight_layout()
plt.show()