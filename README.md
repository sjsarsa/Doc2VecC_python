# Doc2VecC_python
Sklearn compatible python class for Doc2VecC from the paper [Efficient Vector Representation for Documents Through Corruption](https://openreview.net/pdf?id=B1Igu2ogg).

Developed with Python version: 3.6

Uses a custom python module created from the original C implementation https://github.com/mchen24/iclr2017

Also includes python_wrapper for calling the original C code. This C code is not included here but can be downloaded from the repo above.

## Setup

Install c_doc2vecc module with
```
python doc2vecc_extension_setup install
```

## Example usage

```
from doc2vecc import Doc2VecC

# Define training data
documents = ["Human machine interface for lab abc computer applications",
             "Random access memory is volatile memory",
             "Some other random text",
             "Etc etc etc",
             "Random text",
             "Maybe some more text",
             "How much wood would a woodchuck chuck if a woodchuck could chuck wood",
             "Woodchuck would chuck as much wood as a woodchuck could chuck if a woodchuck could chuck wood",
             ]


# Tokenize documents
documents = [doc.split() for doc in documents]

# Train a model
model = Doc2VecC(verbose=0)
model.fit_transform(documents)

# Get similar words
print('Documents most similar to "random woodchuck":')
top5_similar = model.get_similar("random woodchuck", 5)
for i, (doc, score) in enumerate(top5_similar):
    print(f'{i+1}. {documents[doc]} ({score})')
```
