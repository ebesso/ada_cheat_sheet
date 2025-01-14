import spacy, sklearn
import numpy as np
import pandas as pd

####################################################################################################################
#BASIC SPACY

# Load lanugage into nlp
nlp = spacy.load('en_core_web_sm')

# If they want you to load in books
corpus_root = 'books/'
books = list()
import os, codecs
for book_file in os.listdir(corpus_root):
    if ".txt" in book_file:
        print(book_file)
        with codecs.open(os.path.join(corpus_root,book_file),encoding="utf8") as f:
            books.append(f.read())

# Remove new lines (\n is counted otherwise)
books = [" ".join(b.split()) for b in books]

# Apply the nlp pipeline onto a text to get a document object
book = books[0]
doc = nlp(book)

# Return a list of sentences in doc
sentences = [sent for sent in doc.sents]

# Return a list of tokens in doc
tokens = [token.text for token in doc]

# Return list of tuples, .text is the representation and pos_ is the word type
pos_tagged = [(token.text, token.pos_) for token in doc]
print(spacy.explain('CCONJ')) # We can explain a type

# tag_ is an even more detailed version of pos_
pos_tagged = [(token.text, token.tag_) for token in doc]
print(spacy.explain('PRP')) # can also use explain

# Returns a tuple with represenation and label of ents, ents only apply to nouns (substantiv), label_ tries to 
# explains what type of noun it is
[(ent.text, ent.label_) for ent in doc.ents]

# Gives us all stopwords in the english vocab (according to spacy)
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# A way to get the stopwords (or get the other words) in a text
stop_words = [token.text for token in doc if token.is_stop]

# We can access the lemmas of a token with token.lemma, in order to have words that mean the same 
# thing be the same thing 
[token.lemma_ for token in doc if token.text != token.lemma_]

# Chunking (shallow parsin) (check text exercise 1)
for chunk in doc.noun_chunks:
    print(chunk.text)

# Dependency parsing (check text exercise 1)
for token in doc:
    print('Token:',token.text,'Head:',token.head.text, 'Children:',[child for child in token.children])

# Count words (can change text to anything)
from collections import Counter

words = [token.text for token in doc]

# five most common tokens
word_freq = Counter(words)
common_words = word_freq.most_common()

print(common_words)

# We can remove steps in the NLP pipeline in order to save time
nlp.remove_pipe('parser')
nlp.remove_pipe('tagger')

###################################################################################################################



