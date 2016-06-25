import math
from collections import defaultdict
class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCount = defaultdict(lambda: 0)
    self.bigramCount = defaultdict(lambda: defaultdict(lambda: 0))
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    previousWord = ""
    for sentence in corpus.corpus:
        for token in sentence.data:
            word = token.word
            self.unigramCount[word] += 1
            self.bigramCount[previousWord][word] += 1
            previousWord = word

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    previousWord = ""
    score = 0.0
    for word in sentence:
        count = self.bigramCount[previousWord][word]
        tot = self.unigramCount[previousWord]

        if count == 0 or tot == 0:
            t = len(self.unigramCount.items())
            c = self.unigramCount[word] + 1
            v = 2*t
            score += math.log(0.4 * (c/v))
        else:
            score += math.log(count/tot)
        previousWord = word

    return score
