"""
Licence:
Copyright (c) 2014 Emmanuel Benazera
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
import numpy as np

class laplace_confidence:
    """laplace confidence for predicted probabilities
    example:


    probs = [0.33,0.33,0.33]
    lp = laplace_confidence(3)
    cf = lp.compute_confidence(probs)
    print "cf=",cf
    probs = [0.1,0.1,0.8]
    lp = laplace_confidence(3)
    cf = lp.compute_confidence(probs)
    print "cf=",cf
    probs = [ 0.26990365252280046,
              0.07650281713854032,
          0.07187131636930887,
          0.06681801523390264,
          0.06665157413975215,
          0.06594724303350165,
          0.06465198650835942,
          0.06451629349988078,
          0.06354024067659364,
          0.0634382748605426,
          0.06335505037408494,
          0.06280353564273267 ]
    lp = laplace_confidence(12)
    cf = lp.compute_confidence(probs)
    print "cf=",cf

    """
    _mean = -1
    _nclasses = 0
    _reo_probs = []

    def __init__(self, nclasses):
        self._nclasses = nclasses

    def order_for_fit(self, probs):
        probs = sorted(probs)
        if self._nclasses % 2 == 0:
            self._mean = int(self._nclasses / 2)
        else:
            self._mean = int((self._nclasses - 1) / 2)
        self._reo_probs = [0] * self._nclasses
        self._reo_probs[self._mean] = probs[self._nclasses - 1]  # highest prob goes in the middle
        j = -1
        countr = self._mean
        countl = self._mean
        for i in range(self._nclasses - 2, -1, -1):
            if j > 0:
                countr += 1
                self._reo_probs[countr] = probs[i]
            else:
                countl -= 1
                self._reo_probs[countl] = probs[i]
            j *= -1

    def compute_confidence(self, probs):

        if self._nclasses == 2:
            return math.fabs(probs[1] - probs[0])

        self.order_for_fit(probs)
        tconf = 0.0
        nsteps = self._nclasses - 1
        for i in range(1, nsteps):
            dlaplace = self.discrete_laplacian(i)
            tconf += dlaplace
        if self._nclasses == 3:
            tconf /= 2.0
        elif self._nclasses == 4:
            tconf /= 3.0
        elif self._nclasses >= 5:
            tconf /= 4.0
        return tconf

    def discrete_laplacian(self, i):
        return 2.0 * math.fabs(self._reo_probs[i - 1] - self._reo_probs[i])

def compute_confidence(prob_array, nb_class):
    """
    Takes probability and return the resulting confidence
    input: (n * nb_class) array of probabilities
    return: (n * 1) array of confidence score associated with the probability
    """
    lp = laplace_confidence(nb_class)

    confidence = [lp.compute_confidence(prob_array[i, :])
                  for i
                  in range(prob_array.shape[0])]
    # return np.array(confidence).reshape(len(confidence), 1)
    return np.array(confidence).reshape(len(confidence), 1)



# probs = [0.33,0.33,0.33]
# lp = laplace_confidence(3)
# cf = lp.compute_confidence(probs)
#
#
#
# lp = laplace_confidence(3)
#
#
#
# import numpy as np
# ypred = np.random.normal(0.5, 0.1, (10, 3))
#
# # a = [list(ypred[:, i]) for i in range(ypred.shape[1])]
#
# lp.compute_confidence([0.7, 0.2, 0.1])
