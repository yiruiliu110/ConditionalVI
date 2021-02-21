## Data Processing

We have used a corpus of NIPS papers in this tutorial. This corpus contains 1740 documentsbut. If you’re following this tutorial just to learn about LDA I encourage you to consider picking a corpus on a subject that you are familiar with.

See https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html for details


```python
import io
import os.path
import re
import tarfile

import smart_open

def extract_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
    fname = url.split('/')[-1]

    # Download the file to local storage first.
    # We can't read it on the fly because of
    # https://github.com/RaRe-Technologies/smart_open/issues/331
    if not os.path.isfile(fname):
        with smart_open.open(url, "rb") as fin:
            with smart_open.open(fname, 'wb') as fout:
                while True:
                    buf = fin.read(io.DEFAULT_BUFFER_SIZE)
                    if not buf:
                        break
                    fout.write(buf)

    with tarfile.open(fname, mode='r:gz') as tar:
        # Ignore directory entries, as well as files like README, etc.
        files = [
            m for m in tar.getmembers()
            if m.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', m.name)
        ]
        for member in sorted(files, key=lambda x: x.name):
            member_bytes = tar.extractfile(member).read()
            yield member_bytes.decode('utf-8', errors='replace')

docs = list(extract_documents())
```


```python
print(len(docs))
print(docs[0][:500])
```

    1740
    1 
    CONNECTIVITY VERSUS ENTROPY 
    Yaser S. Abu-Mostafa 
    California Institute of Technology 
    Pasadena, CA 91125 
    ABSTRACT 
    How does the connectivity of a neural network (number of synapses per 
    neuron) relate to the complexity of the problems it can handle (measured by 
    the entropy)? Switching theory would suggest no relation at all, since all Boolean 
    functions can be implemented using a circuit with very low connectivity (e.g., 
    using two-input NAND gates). However, for a network that learns a pr
    


```python
# Tokenize the documents.
from nltk.tokenize import RegexpTokenizer

# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]
```


```python
# Lemmatize the documents.
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
```


```python
# Compute bigrams.
from gensim.models import Phrases

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
```


```python
# Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 20% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.2)
```


```python
# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]
```


```python
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
```

    Number of unique tokens: 8013
    Number of documents: 1740
    


```python
from gensim import corpora
```


```python
corpora.MmCorpus.serialize('./nips.mm', corpus)
```


```python
dictionary.save_as_text('./nips_wordids.txt')
```


```python

```


```python

```
