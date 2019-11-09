import os
import numpy as np
from time import time
from sklearn.base import BaseEstimator
import pandas as pd
import util
import \
    c_doc2vecc  # custom module created from Minmin Chen's implementation of DocvecC https://github.com/mchen24/iclr2017
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class Doc2VecC(BaseEstimator):
    """
    Doc2vec with corruption python wrapper for the model from https://github.com/mchen24/iclr2017
    Original paper: https://openreview.net/pdf?id=B1Igu2ogg

    Inherits sklearn BaseEstimator for usage with e.g. sklearn pipeline and GridSearchCV

    Training example: Doc2VecC().fit(['a b c d', 'd c b a', 'x y f a'])
    Embedding example with trained model: model.transform(['a a b b'])

    Get similar example: model.get_similar('a a b b')
                         model.get_similar_by_index(0)
    """

    def __init__(self, train_file='tmp_d2vcc_train_data.txt', wordvec_file='tmp_d2vcc_wordvectors.txt',
                 docvec_file='tmp_d2vcc_docvectors.txt', cbow=True, size=100,
                 window=10, negative=5, sample=0., threads=6, binary=False, epochs=10, min_word_count=0,
                 sentence_sample=0.1, vocab_file='tmp_d2vcc_vocab.txt',
                 keep_generated_files=False, alpha=None, verbose=1):
        """

        :param train_file: Name of file which is used for model training and constructing vocabulary if vocab file is not present or empty.
                            (will be removed after training if keep_generated_files=False)
        :param wordvec_file: Name of file where the trained word vectors i.e. model's input weights are stored.
                            (will be removed after training if keep_generated_files=False)
        :param docvec_file: Name of file where the trained document vectors are stored.
                            (will be removed after training if keep_generated_files=False)
        :param cbow: The model architecture (True: Continuous Bag of Words, False: Skip-gram).
        :param size: Hidden layer size, which defines the resulting vector size of embeddings.
        :param window: Max skip length between words.
        :param negative: Amount of negative samples used in training.
        :param sample: Probability for downsampling common words. (May result in seg fault if > 0 in current C-implementation for small datasets!)
        :param threads: Amount of threads to utilize during training
        :param binary: Save resulting vectors in binary
        :param epochs: Amount of epochs for training
        :param min_word_count: Threshold for discarding infrequent words. Words that appear less than <min_word_count>
                                are excluded.
        :param sentence_sample: The rate to sample words out of a document for document representation
        :param vocab_file: Name of vocabulary file which is used for model training. Required for transform.
                            Will not be removed even with keep_generated_files=True.
        :param keep_generated_files: Determines whether to save train_file, word_vec_file and doc_vec_file
        :param alpha: The starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
        :param verbose: Determine the amount of printing text for the model. 0 or higher.
        """
        self.train_file = train_file
        self.cbow = cbow
        self.size = size
        self.window = window
        self.negative = negative
        self.sample = sample
        self.threads = threads
        self.binary = binary
        self.epochs = epochs
        self.min_word_count = min_word_count
        self.sentence_sample = sentence_sample
        self.vocab_file = vocab_file
        self.keep_generated_files = keep_generated_files
        self.verbose = verbose
        self.wordvec_file = wordvec_file
        self.docvec_file = docvec_file
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = .025 if cbow else .05
        self.docvecs = None
        self.input_weights = None

    def load_from_txt_file(self, path, wordvec_file=None, docvec_file=None,
                           vocab_file=None):
        if path[-1] != '/': path += '/'

        if wordvec_file:
            with open(path + wordvec_file) as f:
                self.size = int(f.readline().split()[1])
            weights = pd.read_csv(path + wordvec_file, skiprows=1, header=None, usecols=range(1, self.size + 1),
                                  delimiter=' ', dtype=np.float32)
            self.input_weights = weights.dropna(axis=1).values
        if vocab_file:
            self.vocab_file = path + vocab_file
        if docvec_file:
            self.docvecs = pd.read_csv(path + docvec_file, header=None, delimiter=' ', dtype=np.float32
                                       ).dropna(axis=1).values
        assert self.docvecs.shape[-1] == self.input_weights.shape[-1], \
            'doc2vec ({}) and wordvec ({}) shapes don\'t match!'.format(self.docvecs.shape[-1],
                                                                        self.input_weights.shape[-1])

    def save_to_txt_file(self, path="", wordvec_file=None, docvec_file=None, vocab_file=None):
        """
        Saves model data to txt files.
        Word vectors are saved in word2vec text format
        """
        if wordvec_file:
            with open(os.path.join(path, vocab_file), 'r') as f:
                words = [line.split(' ')[0] for line in f.read().split('\n')]
            with open(os.path.join(path, wordvec_file), 'w') as f:
                f.write("{} {}\n".format(self.input_weights.shape[0], self.input_weights.shape[1]))
                for word, vector in zip(words, self.input_weights):
                    f.write("{} {}".format(word, " ".join(vector)))
            print("Wrote {}", os.path.join(path, wordvec_file))

    @staticmethod
    def save_to_txt(model, path="", wordvec_file=None, docvec_file=None):
        """
        Saves model data to txt files.
        Word vectors are saved in word2vec text format
        """
        if wordvec_file:
            with open(os.path.join(path, model.vocab_file), 'r') as f:
                words = [line.split(' ')[0] for line in f.read().split('\n')]
            with open(os.path.join(path, wordvec_file), 'w') as f:
                f.write("{} {}\n".format(model.input_weights.shape[0], model.input_weights.shape[1]))
                for word, vector in zip(words, model.input_weights):
                    f.write("{} {}\n".format(word, " ".join([str(e) for e in vector])))
            print("Wrote {}".format(os.path.join(path, wordvec_file)))

    @staticmethod
    def get_docvecs_from_file(filename):
        """
        :param filename: Name of the file containing docvecs.
        :return: Document embeddings as a numpy matrix of floats, one row = one embedding
        """
        with open(filename, 'r') as f:
            docvecs = f.readlines()

        return np.array([list(map(lambda x: float(x), vec.split(' ')[:-1])) for vec in docvecs][:-1])
        # all embeddings in the file end in space, therefore the last elements of the vecs are not included.

    @staticmethod
    def prepare_documents(documents, filename, replace=False):
        """
        c_doc2vecc reads from single file containing documents separated by line breaks.

        """
        if os.path.exists(filename):
            if not replace:
                print('Using already existing file', filename)
            else:
                os.remove(filename)

        with open(filename, 'w') as f:
            tokenized = util.is_tokenized(documents)
            for doc in documents:
                if tokenized:
                    joined_doc = ' '.join(doc)
                    assert ('\n' not in joined_doc)
                    f.write(joined_doc + '\n')
                else:
                    assert ('\n' not in doc)
                    f.write(doc + '\n')

    def generate_vocabulary(self, documents, replace=False):
        """
        :param documents:
        :param replace: Determine whether to overwrite existing vocabulary
        :return:
        """

        self.prepare_documents(documents, self.train_file, self.keep_generated_files)
        if replace or not os.path.exists(self.vocab_file):
            if os.path.exists(self.vocab_file): os.remove(self.vocab_file)
            c_doc2vecc.generate_vocab_file(self.train_file, self.vocab_file,
                                           min_count=self.min_word_count, debug_mode=self.verbose)

    def fit(self, documents=None, y=None):
        """
        :param documents: list of tokenized documents
        :return: self
        """

        self.generate_vocabulary(documents, replace=True)
        start = time()

        cbow = 1 if self.cbow else 0
        binary = 1 if self.binary else 0
        if not self.keep_generated_files:
            self.input_weights, self.docvecs = c_doc2vecc.train(size=self.size, train_file=self.train_file,
                                                                test_file=self.train_file, window=self.window,
                                                                read_vocab=self.vocab_file, debug_mode=self.verbose,
                                                                min_count=self.min_word_count, sample=self.sample,
                                                                negative=self.negative, num_threads=self.threads,
                                                                cbow=cbow, binary=binary, iter=self.epochs,
                                                                rp_sample=self.sentence_sample, alpha=self.alpha)
            os.remove(self.train_file)
        else:
            self.input_weights, self.docvecs = c_doc2vecc.train(size=self.size, train_file=self.train_file,
                                                                test_file=self.train_file, window=self.window,
                                                                min_count=self.min_word_count, sample=self.sample,
                                                                negative=self.negative, num_threads=self.threads,
                                                                cbow=self.cbow, binary=self.binary, iter=self.epochs,
                                                                rp_sample=self.sentence_sample, alpha=self.alpha,
                                                                read_vocab=self.vocab_file, debug_mode=self.verbose,
                                                                docvec_out=self.docvec_file,
                                                                wordvec_out=self.wordvec_file)
        util.print_elapsed('model trained', start, format='min')
        return self

    def embed(self, X):
        """
        :param X:
        :return:
        """
        start = time()
        tmp_transform_file = 'tmp_d2vcc_transform_file.txt'
        if self.verbose > 0:
            print('creating file', tmp_transform_file)
            print('starting document embedding')
        self.prepare_documents(X, tmp_transform_file, self.keep_generated_files)
        if not self.keep_generated_files:
            docvecs = c_doc2vecc.transform(weight_file=self.wordvec_file,
                                           train_file=tmp_transform_file, test_file=tmp_transform_file,
                                           debug_mode=self.verbose,
                                           weights=self.input_weights, size=self.size,
                                           min_count=self.min_word_count, sample=self.sample,
                                           read_vocab=self.vocab_file)
            os.remove(tmp_transform_file)
            if self.verbose > 0: print('tmp files removed')
        else:
            docvecs = c_doc2vecc.transform(weight_file=self.wordvec_file,
                                           train_file=tmp_transform_file, test_file=tmp_transform_file,
                                           debug_mode=self.verbose,
                                           weights=self.input_weights, size=self.size,
                                           min_count=self.min_word_count, sample=self.sample,
                                           docvec_out=self.docvec_file,
                                           read_vocab=self.vocab_file)

        if self.verbose > 0: util.print_elapsed('documents embedded', start, format='s')
        return docvecs

    def infer_vector(self, text):
        return self.embed([text])[0]

    def transform(self, documents=None, y=None):
        """
        :param documents: documents to be embedded
        :return: document vectors
        """
        if documents is None:
            return self.docvecs
        else:
            return self.embed(documents)

    def fit_transform(self, documents=None, y=None):
        self.fit(documents=documents)
        return self.docvecs

    def remove_created_files(self):
        for filename in (self.train_file, self.wordvec_file, self.docvec_file):
            if os.path.exists(filename): os.remove(filename)

    def get_similar(self, query, topn=100):
        query_vec = self.infer_vector(query)
        sims = cosine_similarity(query_vec.reshape(1, -1), self.docvecs)[0]
        return util.get_topn_with_indices(sims, topn, offset=0)

    def get_similar_by_index(self, index, topn=100):
        i = int(index)
        sims = cosine_similarity(self.docvecs[i].reshape(1, -1), self.docvecs)[0]
        return util.get_topn_with_indices(sims, topn, offset=1)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
