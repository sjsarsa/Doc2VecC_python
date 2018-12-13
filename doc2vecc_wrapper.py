import os
from subprocess import run, PIPE
import numpy as np
from time import time
from sklearn.base import BaseEstimator


class Doc2VecC(BaseEstimator):
    """
    Doc2vec with corruption python wrapper for the model from https://github.com/mchen24/iclr2017
    Original paper: https://openreview.net/pdf?id=B1Igu2ogg
    """

    def __init__(self, train_file='train_data.txt', test_file=None, word_out='wordvectors.txt', doc_out='docvectors.txt',
                 cbow=1, size=100,
                 window=10, negative=5, hs=0, sample=0, threads=12, binary=0, iter=20, min_count=0,
                 sentence_sample=0.1, save_vocab='d2vC.data.vocab', doc2vecc_exec_file='doc2vecc',
                 generate_suffix=False, generate_train_file=True, dir_path=''):
        """
        Parameters
        ----------
        :param doc2vecc_exec_file: Filename or path of the compiled C implementation of doc2vecc
        :param generate_suffix: Boolean to determine whether a filename suffix is generated from given parameters
                                The resulting suffix is '_D{size}_win{window}_neg{negative}_iter{}.txt'
        :param output_docs_file: Filename or path to data containing the documents that will be embedded by the model
                                 Corresponds to the undocumented '-test' parameter in the C-implementation.

        Parameters for training: (from the C implementation)
        ---------------------------------------------------

        train_file -- train <file>
            Use text data from <file> to train the model
        word_out -- word <file>
            Use <file> to save the resulting word vectors
        doc_out -- output <file>
            Use <file> to save the resulting document vectors
        size <int>
            Set size of word vectors; default is 100
        window <int>
            Set max skip length between words; default is 5
        sample <float>
            Set threshold for occurrence of words. Those that appear with higher frequency in the training data
            will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        hs <int>
            Use Hierarchical Softmax; default is 0 (not used)
        negative <int>
            Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
        threads <int>
            Use <int> threads (default 12)
        iter <int>
            Run more training iterations (default 10)
        min-count <int>
            This will discard words that appear less than <int> times; default is 5
        alpha <float>
            Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
        debug <int>
            Set the debug mode (default = 2 = more info during training)
        binary <int>
            Save the resulting vectors in binary moded; default is 0 (off)
        save-vocab <file>
            The vocabulary will be saved to <file>
        read-vocab <file>
            The vocabulary will be read from <file>, not constructed from the training data
        cbow <int>
            Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)
        sentence-sample <float>
            The rate to sample words out of a document for document representation
        \nExamples:
        ./doc2vecc -train data.txt -output docvec.txt -word wordvec.txt -sentence-sample 0.1 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n
        """

        if dir_path[-1] != '/': dir_path += '/'

        self.train_file = dir_path + train_file
        self.test_file = dir_path + test_file if test_file else dir_path + train_file
        self.cbow = cbow
        self.size = size
        self.window = window
        self.negative = negative
        self.hs = hs
        self.sample = sample
        self.threads = threads
        self.binary = binary
        self.iter = iter
        self.min_count = min_count
        self.sentence_sample = sentence_sample
        self.save_vocab = save_vocab
        self.doc2vecc_exec_file = dir_path + doc2vecc_exec_file
        self.generate_train_file = generate_train_file
        self.docvecs = None
        if generate_suffix:
            self.word_out = '{}_D{}_win{}_neg{}_iter{}.txt'.format(word_out, size, window, negative, iter)
            self.doc_out = '{}_D{}_win{}_neg{}_iter{}.txt'.format(doc_out, size, window, negative, iter)
        else:
            self.word_out = word_out
            self.doc_out = doc_out

    # data IO
    @staticmethod
    def prepare_document_file_for_doc2vecc(documents, filename):
        """
        doc2vec.c takes as input a single file containing documents separated by line breaks.

        Creates a formatted text file for doc2vec.c train input parameter
        """

        def is_tokenized(x):
            """
            TODO: error handling and maybe put this func into some text util module
            """
            if isinstance(x[0], str): return False
            if isinstance(x[0][0], str): return True

        print('train file', filename)
        try:
            os.remove(filename)
        except OSError:
            pass
        with open(filename, 'w') as f:
            tokenized = is_tokenized(documents)
            for doc in documents:
                if tokenized:
                    line_breakless_doc = list(filter(lambda x: x not in ['\r', '\n', '\\n', '\\\n'], doc))
                    f.write(' '.join([w.rstrip('\r\n\\').replace('\\n', '') for w in line_breakless_doc]) + '\n')
                else:
                    line_breakless_doc = doc.replace('\n', ' ')
                    f.write(line_breakless_doc + '\n')

    @staticmethod
    def get_docvecs_from_file(filename):
        """

        :param filename: name of the file containing docvecs. Same as 'doc_out' param for the model
        :return: Document embeddings as a numpy matrix of floats, one row = one embedding
        """
        with open(filename, 'r') as f:
            docvecs = f.readlines()

        return np.array([list(map(lambda x: float(x), vec.split(' ')[:-1])) for vec in docvecs][:-1])
        # all embeddings in the file end in space, therefore the last (empty) elements  of the vecs are excluded.

    def fit(self, documents=None, y=None):
        """

        :param documents: list of tokenized documents
        :param train_file: filename for saving or reading formatted documents for doc2vecc.
        If documents are provided, the formatted documents are saved to train file, other train file is read and must
        exist
        :param tokenized:
        :return: None

        embeddings are saved in files specified in the doc_out and word_out parameters for the model
        """
        if not documents or not self.generate_train_file:
            assert os.path.exists(self.train_file)
        else:
            print('processing documents and saving data in ', self.train_file)
            self.prepare_document_file_for_doc2vecc(documents, self.train_file)
        print(self.train_file)
        print(self.doc2vecc_exec_file)
        assert os.path.exists(self.doc2vecc_exec_file)
        assert os.path.exists(self.test_file)
        if os.path.exists(self.word_out): os.remove(self.word_out)
        if os.path.exists(self.doc_out): os.remove(self.doc_out)

        start = time()

        """
        The test parameter needs to be the original data file, the docvecs are saved by using the test file. 
        In the 'go.sh' script provided by the authors of docvecc, the train data is shuffled and 
        the test data is the unshuffled data.
        Here the train file is the same as test file, it seemed to perform better than doc2vec without shuffling

        TODO: test the effect of shuffling
        """
        doc2vec_c_cmd = " ".join(
            list(map(lambda x: str(x),
                     ['./' + self.doc2vecc_exec_file, '-train', self.train_file, '-word', self.word_out, '-output',
                      self.doc_out, '-sentence-sample', self.sentence_sample, '-size', self.size,
                      '-window', self.window, '-min-count', self.min_count, '-threads', self.threads,
                      '-sample', self.sample, '-negative', self.negative, '-hs', self.hs, '-binary', self.binary,
                      '-cbow', self.cbow, '-iter', self.iter, '-test', self.test_file])))

        print('executing command:', doc2vec_c_cmd)
        proc = run(doc2vec_c_cmd, shell=True, stdout=PIPE, check=True)
        print(proc)
        print('docvecs generated in {:.5} min'.format((time() - start) / 60))
        self.docvecs = self.get_docvecs_from_file(self.doc_out)
        return self

    def get_docvecs(self):
        return self.get_docvecs_from_file(self.doc_out)

    def transform(self, X=None):
        """
        For sklearn pipeline, does not actually transform inputs, since it is not implemented in the C-code

        :param X: dummy variable 
        :return: Already transformed document vectors from "output_docs_file"
        """
        return self.get_docvecs()

    def fit_transform(self, documents=None, y=None):
        self.fit(documents=documents)
        return self.transform()
