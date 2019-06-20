import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

train = pd.read_csv('./data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

test = pd.read_csv('./data/testData.tsv', header=0, delimiter='\t', quoting=3)


example1 = BeautifulSoup(train['review'][0], "html5lib")

letters_only = re.sub('[^a-zA-Z]', ' ', example1.get_text())

lower_case = letters_only.lower()

words = lower_case.split()

words = [w for w in words if not w in stopwords.words('english')]

stemmer = SnowballStemmer('english')

words = [stemmer.stem(w) for w in words]

wordnet_lemaatizer = WordNetLemmatizer()

words = [wordnet_lemaatizer.lemmatize(w) for w in words]

print(words)

 