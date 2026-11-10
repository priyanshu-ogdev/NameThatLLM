import numpy as np
import scipy.sparse as sp
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class DummyDense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.ones((len(X), 10), dtype=np.float64)

fu = FeatureUnion([
    ('tfidf', TfidfVectorizer(max_features=100)),
    ('dense', DummyDense())
])
texts = ['hello world this is a test', 'another text example here']
res = fu.fit_transform(texts)
print('res type:', type(res))
print('is sparse:', sp.issparse(res))
if sp.issparse(res):
    print('Sparse format:', res.format)

