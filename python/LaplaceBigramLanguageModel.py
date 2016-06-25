import math
from collections import defaultdict
class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCount = defaultdict(lambda:0)
    self.bigramCount = defaultdict(lambda: defaultdict(lambda:0))
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
    score = 0.0
    previousWord = ""
    v = len(self.unigramCount.items())
    for word in sentence:
        count = self.bigramCount[previousWord][word] + 1
        tot = self.unigramCount[word] + v
        score += math.log(count/tot)
        previousWord = word
    return score
