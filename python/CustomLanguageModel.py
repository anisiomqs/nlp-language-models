import math
from collections import defaultdict

class CustomLanguageModel:
  # Kneser-ney implementation - score of ~0.25
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.total = 0
    self.reverseBigramCount = defaultdict(lambda : defaultdict(lambda : 0))
    self.bigramCount = defaultdict(lambda : defaultdict(lambda : 0))
    self.unigramCount = defaultdict(lambda: 0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    lastToken = "#"
    for sentence in corpus.corpus:
      for datum in sentence.data:
        token = datum.word
        self.reverseBigramCount[token][lastToken] += 1
        self.bigramCount[lastToken][token] += 1
        self.unigramCount[token] += 1
        self.total += 1
        lastToken = token

  def delta(self,count):
    if count > 2:
      return 0.75
    else:
      return 0.5

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    lastToken = "#"

    for token in sentence:
      bigramCount = self.bigramCount[lastToken][token] + 1e-6
      lastTokenCount = self.unigramCount[lastToken]
      d = self.delta(bigramCount)

      if (lastTokenCount == 0 ):
        lastTokenCount = 999999

      # equivalent to: |{ w : c(w_i-1,w) > 0}|
      bigramsInitiatedByPreviousToken = len(self.bigramCount[lastToken].items()) + 0.001

      #                     d
      # lambda (w_i-1) = -------- |{ w : c(w_i-1, w) > 0}|
      #                  c(w_i-1)
      l = (d / lastTokenCount) * bigramsInitiatedByPreviousToken

      #                    |{ w_i-1 : c(w_i-1,w_i) > }|       => Bigrams terminated in w_i
      # P            (w_i) = ------------------------------
      #  continuation      |{ w_j-1 : c(w_j-1, w_j) > 0}|     => Total number of bigram types
      continuationProb = float(len(self.reverseBigramCount[token].items())) / self.total

      score += math.log(bigramCount/lastTokenCount + l * continuationProb)

      lastToken = token
    return score
