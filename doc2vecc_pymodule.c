// Copyright 2017 Criteo Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// This file has been modified into a python module with some refactoring. No changes to the algorithm are made.
// Orignal paper: Minmin Chen. "Efficient Vector Representation for Documents Through Corruption." 5th International Conference on Learning Representations, ICLR (2017)
// Original C implementation: https://github.com/mchen24/iclr2017

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define MAX_STRING 100
#define EXP_TABLE_SIZE 2000
#define MAX_EXP 10
#define MAX_SENTENCE_SAMPLE 100
#define MAX_SENTENCE_LENGTH 10000
#define MAX_CODE_LENGTH 40

const int VOCAB_HASH_SIZE = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

void InitUnigramTable(struct vocab_word *vocab, long long vocab_size, int **table, int table_size) {
  int a, i;
  long long train_words_pow = 0;
  float d1, power = 0.75;
  *table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (float)train_words_pow;
  for (a = 0; a < table_size; a++) {
    (*table)[a] = i;
    if (a / (float)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (float)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % VOCAB_HASH_SIZE;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word, struct vocab_word *vocab, int *vocab_hash) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % VOCAB_HASH_SIZE;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, struct vocab_word *vocab, int *vocab_hash) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word, vocab, vocab_hash);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, struct vocab_word **vocabP, int *vocab_hash, long long *vocab_sizeP, long long *vocab_max_sizeP) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  (*vocabP)[*vocab_sizeP].word = (char *)calloc(length, sizeof(char));
  strcpy((*vocabP)[*vocab_sizeP].word, word);
  (*vocabP)[*vocab_sizeP].cn = 0;
  *vocab_sizeP += 1;
  // reallocate memory if needed
  if (*vocab_sizeP + 2 >= *vocab_max_sizeP) {
    *vocab_max_sizeP *= 2;
    *vocabP = (struct vocab_word *)realloc(*vocabP, *vocab_max_sizeP * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
  vocab_hash[hash] = *vocab_sizeP - 1;
  return *vocab_sizeP - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(struct vocab_word **vocabP, int *vocab_hash,
               long long *vocab_sizeP, long long *train_wordsP, int min_count) {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&(*vocabP)[1], *vocab_sizeP - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < VOCAB_HASH_SIZE; a++) vocab_hash[a] = -1;
  size = *vocab_sizeP;
  *train_wordsP = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((*vocabP)[a].cn < min_count) {
      *vocab_sizeP-=1;
      free((*vocabP)[*vocab_sizeP].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash((*vocabP)[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
      vocab_hash[hash] = a;
      *train_wordsP += (*vocabP)[a].cn;
    }
  }
  *vocabP = (struct vocab_word *)realloc(*vocabP, (*vocab_sizeP + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < *vocab_sizeP; a++) {
    (*vocabP)[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    (*vocabP)[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct vocab_word *vocab, int *vocab_hash, long long *vocab_sizeP, int *min_reduceP) {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < *vocab_sizeP; a++) if (vocab[a].cn > *min_reduceP) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  *vocab_sizeP = b;
  for (a = 0; a < VOCAB_HASH_SIZE; a++) vocab_hash[a] = -1;
  for (a = 0; a < *vocab_sizeP; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  *min_reduceP += 1;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree(struct vocab_word *vocab, long long vocab_size) {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile(char *train_file, long long *train_wordsP, int debug_mode, struct vocab_word **vocabP, int *vocab_hash,
                             long long *vocab_sizeP, long long *vocab_max_sizeP, int* min_reduceP, long long *file_sizeP,
                             int min_count) {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < VOCAB_HASH_SIZE; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("training ERROR: data file not found!\n");
    exit(1);
  }
  *vocab_sizeP = 0;
  AddWordToVocab((char *)"</s>", vocabP, vocab_hash, vocab_sizeP, vocab_max_sizeP);
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    *train_wordsP+=1;
    if ((debug_mode > 1) && (*train_wordsP % 1000000 == 0)) {
      printf("%lldK%c", *train_wordsP / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word, *vocabP, vocab_hash);
    if (i == -1) {
      a = AddWordToVocab(word, vocabP, vocab_hash, vocab_sizeP, vocab_max_sizeP);
      (*vocabP)[a].cn = 1;
    } else (*vocabP)[i].cn+=1;
    if (*vocab_sizeP > VOCAB_HASH_SIZE * 0.7) ReduceVocab(*vocabP, vocab_hash, vocab_sizeP, min_reduceP);
  }
  SortVocab(vocabP, vocab_hash, vocab_sizeP, train_wordsP, min_count);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", *vocab_sizeP);
    printf("Words in train file: %lld\n", *train_wordsP);
  }
  *file_sizeP = ftell(fin);
  fclose(fin);
}

void SaveVocab(char *save_vocab_file, struct vocab_word *vocab, long long vocab_size) {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab(char *train_file, char *read_vocab_file, struct vocab_word **vocabP, int *vocab_hash, long long *vocab_sizeP, long long *vocab_max_sizeP,
               long long *train_wordsP, long long *file_sizeP, int debug_mode, int min_count) {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin;
  fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < VOCAB_HASH_SIZE; a++) vocab_hash[a] = -1;
  *vocab_sizeP = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word, vocabP, vocab_hash, vocab_sizeP, vocab_max_sizeP);
    fscanf(fin, "%lld%c", &(*vocabP)[a].cn, &c);
    i++;
  }
  SortVocab(vocabP, vocab_hash, vocab_sizeP, train_wordsP, min_count);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", *vocab_sizeP);
    printf("Words in vocab file: %lld\n", *train_wordsP);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  *file_sizeP = ftell(fin);
  fclose(fin);
}

int countLines(char *file) {
    int count = 0;
    char c;
    FILE *fp = fopen(file, "rb");
    if (fp == NULL) {
         printf("File '%s' not found\n", file);
         exit(1);
    }
    for (c = getc(fp); c != EOF; c = getc(fp)) {
        if (c == '\n') count++;
    }
    fclose(fp);
    return count;
}

struct thread_params {
    long id;
    char **train_file;
    float **syn0;
    float **syn1;
    float **syn1neg;
    float **expTable;
    struct vocab_word **vocab;
    int **vocab_hash;
    int binary;
    int cbow;
    int debug_mode;
    int window;
    int num_threads;
    int min_count;
    long long vocab_max_size;
    long long vocab_size;
    long long layer1_size;
    long long train_words;
    long long *word_count_actual;
    long long file_size;
    long long iter;
    float *alpha;
    float starting_alpha;
    float sample;
    float rp_sample;
    int negative;
    int **table;
    int table_size;
    clock_t start;
};

void *TrainModelThread(void *parameters) {
  struct thread_params *params = parameters;
  int negative = params->negative;
  char *train_file = *params->train_file;
  float *syn1neg = *params->syn1neg;
  float *expTable = *params->expTable;
  int table_size = params->table_size;
  float rp_sample = params->rp_sample;
  float sample = params->sample;
  float starting_alpha = params->starting_alpha;
  float *alpha = params->alpha;
  int cbow = params->cbow;
  int window = params->window;
  int *vocab_hash = *params->vocab_hash;
  int debug_mode = params->debug_mode;
  int num_threads = params->num_threads;
  struct vocab_word *vocab = *params->vocab;
  clock_t start = params->start;
  long long vocab_size = params->vocab_size;
  long long train_words = params->train_words;
  long long *word_count_actual = params->word_count_actual;
  long long iter = params->iter;
  long long file_size = params->file_size;
  long long layer1_size = params->layer1_size;
  int *table = *params->table;
  float *syn0 = *params->syn0;
  float *syn1 = *params->syn1;


  long long a, b, d, t, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1], sen_sample[MAX_SENTENCE_SAMPLE];
  long long l1, l2, c, target, label, local_iter = iter;
  int id = params->id;

  unsigned long long next_random = (long long)id;
  float f, g, w;
  clock_t now;
  float *neu1 = (float *)calloc(layer1_size, sizeof(float));
  float *neu1e = (float *)calloc(layer1_size, sizeof(float));

  FILE *fi = fopen(train_file, "rb");
  if (fi == NULL) {
    printf("file %s not found\n", train_file);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
  }
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      *word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, *alpha,
         *word_count_actual / (float)(iter * train_words + 1) * 100,
         *word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      *alpha = starting_alpha * (1 - *word_count_actual / (float)(iter * train_words + 1));
      if (*alpha < starting_alpha * 0.0001) *alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi, vocab, vocab_hash);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          float ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (float)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > train_words / num_threads)) {
      *word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      // MC: it's not neccesary an even window, the starting point is random.
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
        // handle the sentence using Doc2VecC
        for (c = 0; c < MAX_SENTENCE_SAMPLE; c ++) { sen_sample[c] = -1; }
        int already_sampled = 0;
        for (t = 0; t < sentence_length; t ++) {
                // randomly sample rp_sample ((0, 1]) of the words in the sentence to represent one sentence
                next_random = next_random * (unsigned long long)25214903917 + 11;
                if ((next_random & 0xFFFF) / (float)65536 > rp_sample) continue;
                //if (t == sentence_position) continue;
                last_word = sen[t];
                if (last_word == -1) continue;
                sen_sample[already_sampled] = last_word;
                already_sampled ++;
                if (already_sampled >= MAX_SENTENCE_SAMPLE) break;
        }
        w = 1.0 / rp_sample / sentence_length;
        for (t = 0; t < already_sampled; t ++) {
                l1 = sen_sample[t] * layer1_size;
                for (c = 0; c < layer1_size; c ++) neu1[c] += w * syn0[c + l1];
        }
        cw ++;

      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * *alpha;
          else if (f < -MAX_EXP) g = (label - 0) * *alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * *alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c] / cw;
        }
    // backprop to the words selected to represent the sentence
        w = 1.0 / rp_sample / sentence_length;
        for (t = 0; t < already_sampled; t ++) {
                l1 = sen_sample[t] * layer1_size;
        for (c = 0; c < layer1_size; c ++) syn0[c + l1] += neu1e[c] * w / cw;
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {

        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1[c] = syn0[c + l1];

        // handle the sentence using Doc2VecC
        for (c = 0; c < MAX_SENTENCE_SAMPLE; c ++) { sen_sample[c] = -1; }
        int already_sampled = 0;
        for (t = 0; t < sentence_length; t ++) {
        // randomly sample rp_sample ((0, 1]) of the words in the sentence to represent one sentence
                next_random = next_random * (unsigned long long)25214903917 + 11;
                if ((next_random & 0xFFFF) / (float)65536 > rp_sample) continue;
                if (t == sentence_position) continue;
                last_word = sen[t];
                if (last_word == -1) continue;
                sen_sample[already_sampled] = last_word;
                already_sampled ++;
                if (already_sampled >= MAX_SENTENCE_SAMPLE) break;
        }
        w = 1.0 / rp_sample / sentence_length;
        for (t = 0; t < already_sampled; t ++) {
                l1 = sen_sample[t] * layer1_size;
                //w = 1.0 / already_sampled;
                for (c = 0; c < layer1_size; c ++) neu1[c] += w * syn0[c + l1];
        }

        for (c = 0; c < layer1_size; c++) neu1[c] /= 2;

        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * *alpha;
          else if (f < -MAX_EXP) g = (label - 0) * *alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * *alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c]/2;
        // backprop to the words selected to represent the sentence
        w = 1.0 / rp_sample / sentence_length;
        for (t = 0; t < already_sampled; t ++) {
                l1 = sen_sample[t] * layer1_size;
                //w = 1.0/already_sampled;
                 for (c = 0; c < layer1_size; c ++) syn0[c + l1] += neu1e[c] * w/2;
        }
      }

    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);

  pthread_exit(NULL);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

static PyArrayObject* embed_docs(char* test_file, char* output_file, int layer1_size, float sample, long long train_words,
                                 struct vocab_word *vocab, int* vocab_hash, PyArrayObject *weights) {
    int ndocs = countLines(test_file);
    npy_intp dv_dims[2] = {ndocs, layer1_size};
    PyArrayObject* docvecs = PyArray_SimpleNew(2, dv_dims, NPY_FLOAT);

    long long t, word, sentence_length = 0,  word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1;
    long long next_random = (long long) 12;
    float w; int c; int docvecs_i = 0;
    float *neu1 = (float *)calloc(layer1_size, sizeof(float));
    FILE *fi = fopen(test_file, "rb");
    FILE *fo;
    if (output_file != NULL) fo = fopen(output_file, "wb");

    while (!feof(fi)) {
        sentence_length = 0;
        while (1) {
            word = ReadWordIndex(fi, vocab, vocab_hash);
            if (feof(fi)) break;
            if (word == -1) continue;
            word_count++;
            if (word == 0) break;
            // The subsampling randomly discards frequent words while keeping the ranking same
            if (sample > 0) {
                float ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                if (ran < (next_random & 0xFFFF) / (float)65536) continue;
            }
            if (sentence_length < MAX_SENTENCE_LENGTH) {
                    sen[sentence_length] = word;
                    sentence_length++;
            }
        }
        if (docvecs_i >= ndocs) continue;
        for (c = 0; c < layer1_size; c ++) neu1[c] = 0;
        for (t = 0; t < sentence_length; t ++ ) {
            if (sen[t] == -1) continue;
            l1 = sen[t];
            w = 1.0/sentence_length;
            for (c = 0; c < layer1_size; c ++) neu1[c] += w * *((float *)PyArray_GETPTR2(weights, l1, c));
        }
        if (docvecs_i < ndocs) for (c = 0; c < layer1_size; c++) {
            if (output_file != NULL) fprintf(fo, "%lf ", neu1[c]);
            *((float *)PyArray_GETPTR2(docvecs, docvecs_i, c)) = neu1[c];
        }
        docvecs_i++;
        if (output_file != NULL) fprintf(fo, "\n");
    }
    fclose(fi);
    if (output_file != NULL) fclose(fo);
    free(neu1);

    return docvecs;
}

static PyObject *train(PyObject *self, PyObject *args, PyObject *kws) {
    import_array();

    struct thread_params params;
    PyObject *train_file_py, *test_file_py;
    char *train_file, *test_file;
    char *read_vocab_file = NULL;
    char *output_file = NULL;
    char *wordembedding_file = NULL;

    params.train_file = &train_file;

    struct vocab_word *vocab;
    params.vocab = &vocab;

    int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
    int *vocab_hash; params.vocab_hash = &vocab_hash;
    long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
    long long train_words = 0, word_count_actual = 0, iter = 10, file_size = 0;
    float alpha = 0.025, starting_alpha, sample = 1e-3, rp_sample=0.1;
    float *syn0, *syn1, *syn1neg, *expTable;
    params.syn0 = &syn0; params.syn1 = &syn1; params.syn1neg = &syn1neg; params.expTable = &expTable;
    clock_t start;

    // the part of code on hs is not working -> is removed
    int negative = 5;
    const int table_size = 1e8;
    int *table;
    params.table_size = table_size;
    params.table = &table;
    int i;
    static char *kwlist[] = {
        "size", "train_file", "test_file", "docvec_out", "wordvec_out", "read_vocab",
          "debug_mode", "binary", "cbow", "alpha", "window", "sample", "negative", "num_threads", "iter",
          "min_count", "rp_sample", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
      args, kws, "iss|OOsiiififiiiif", kwlist,
      &layer1_size,        //i
      &train_file,         //s
      &test_file,          //s
//    ------optional------
      &output_file,        //s
      &wordembedding_file, //s
      &read_vocab_file,    //s
      &debug_mode,         //i
      &binary,             //i
      &cbow,               //i
      &alpha,              //f
      &window,             //i
      &sample,             //f
      &negative,           //i
      &num_threads,        //i
      &iter,               //i
      &min_count,          //i
      &rp_sample           //f
    ))
    if (cbow) alpha = 0.05;

    if (debug_mode > 1) {
        printf("params: size=%d, cbow=%d, alpha=%f, window=%d, negative=%d, num_threads=%d, iter=%d, rp_sample=%f, min_count=%d, sample=%f\n",
                layer1_size, cbow, alpha, window, negative, num_threads, iter, rp_sample, min_count, sample);
        printf("doc vec out file: %s \n", output_file);
        printf("word embed file: %s read vocab file %s\n", wordembedding_file, read_vocab_file);
    }
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(VOCAB_HASH_SIZE, sizeof(int));

    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

    long a, b;
    FILE *fo, *fo_w;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    if (read_vocab_file != NULL) {
        FILE *ft = fopen(read_vocab_file, "rb");
        if (ft != NULL) {
            fclose(ft);
            ReadVocab(train_file, read_vocab_file, &vocab, vocab_hash, &vocab_size, &vocab_max_size,
                                           &train_words, &file_size, debug_mode, min_count);
        }
    }
    if (vocab_size == 0) LearnVocabFromTrainFile(train_file, &train_words, debug_mode, &vocab, vocab_hash, &vocab_size, &vocab_max_size,
                                 &min_reduce, &file_size, min_count);

    // initialize neural network
    unsigned long long next_random = 1;
    // one weight for representation, one for weighting
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
    }

    CreateBinaryTree(vocab, vocab_size);

    InitUnigramTable(vocab, vocab_size, &table, table_size);
    start = clock();
    starting_alpha = alpha;
    word_count_actual = 0;

    params.alpha = &alpha;
    params.layer1_size = layer1_size;
    params.binary = binary;
    params.cbow = cbow;
    params.window = window;
    params.sample = sample;
    params.negative = negative;
    params.num_threads = num_threads;
    params.iter = iter;
    params.min_count = min_count;
    params.rp_sample = rp_sample;
    params.starting_alpha = starting_alpha;
    params.word_count_actual = &word_count_actual;
    params.start = start;
    params.vocab_size = vocab_size; params.vocab_max_size = vocab_max_size;
    params.file_size = file_size; params.train_words = train_words;


    if (debug_mode > 0) printf("starting embedding training\n");
    for (a = 0; a < num_threads; a++) {
        struct thread_params new_params = params;
        new_params.id = a;
        pthread_create(&pt[a], NULL, TrainModelThread, (void *)&new_params);
    }
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    if (debug_mode > 0) printf("finished embedding training\n");

    if(wordembedding_file != NULL) {
      if (debug_mode > 0) printf("saving word vectors to %s\n", wordembedding_file);
      fflush(stdout);

      // Save the word vectors
      fo_w = fopen(wordembedding_file, "wb");
      if (debug_mode > 0) fprintf(fo_w, "%lld %lld\n", vocab_size, layer1_size);
      for (a = 0; a < vocab_size; a++) {
        fprintf(fo_w, "%s ", vocab[a].word);
            if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo_w);
            else for (b = 0; b < layer1_size; b++) fprintf(fo_w, "%lf ", syn0[a * layer1_size + b]);
            fprintf(fo_w, "\n");
      }
      if (wordembedding_file != NULL) fclose(fo_w);
    }

    long w_dims[2] = {vocab_size, layer1_size};
    PyArrayObject* w2v_weights = PyArray_SimpleNewFromData(2, w_dims, NPY_FLOAT, (void *)syn0);

    PyArrayObject* docvecs = embed_docs(test_file, output_file, layer1_size, sample, train_words, vocab, vocab_hash,
                                        w2v_weights);

    free(table);
    free(expTable);
    for (a = 0; a < vocab_size; a++) {
        free(vocab[a].word);
        free(vocab[a].code);
        free(vocab[a].point);
    }
    free(pt);
    free(vocab); free(vocab_hash);

    free(syn1neg);

    PyObject *result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, w2v_weights);
    PyTuple_SetItem(result, 1, docvecs);

    PyArray_ENABLEFLAGS((PyArrayObject*)w2v_weights, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject*)docvecs, NPY_ARRAY_OWNDATA);
    return result;
}

static PyObject *transform(PyObject *self, PyObject *args, PyObject *kws) {
    import_array();

    char *weight_file, *train_file, *test_file;
    char *output_file = NULL;
    char *read_vocab_file = NULL;
    struct vocab_word *vocab;
    int debug_mode = 2, min_count = 5, min_reduce = 1;
    int *vocab_hash;
    long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
    long long train_words = 0, file_size = 0;
    float sample = 1e-3;
    PyArrayObject *weights = NULL;

    static char *kwlist[] = {
        "weight_file", "train_file", "test_file", "docvec_out", "read_vocab", "debug_mode", "min_count",
         "sample", "weights", "size", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
      args, kws, "sss|ssiifOi", kwlist,
      &weight_file,        //s
      &train_file,         //s
      &test_file,          //s
//    ------ optional ------
      &output_file,        //s
      &read_vocab_file,    //s
      &debug_mode,         //i
      &min_count,          //i
      &sample,             //f
      &weights,            //O
      &layer1_size         //i
    ))
    if (debug_mode > 1) {
        printf("params: size=%lld, min_count=%d, sample=%f\n",
                                layer1_size, min_count, sample);
        printf("weight file %s train file %s, test file %s, doc vec out file: %s \n", weight_file, train_file, test_file, output_file);
        printf("read vocab file %s\n", read_vocab_file);
    }

    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(VOCAB_HASH_SIZE, sizeof(int));

    if (read_vocab_file != NULL) {
        FILE *ft = fopen(read_vocab_file, "rb");
        if (ft != NULL) {
            fclose(ft);
            ReadVocab(train_file, read_vocab_file, &vocab, vocab_hash, &vocab_size, &vocab_max_size,
                                        &train_words, &file_size, debug_mode, min_count);
        }
     }
    if (vocab_size == 0) LearnVocabFromTrainFile(train_file, &train_words, debug_mode, &vocab, vocab_hash, &vocab_size, &vocab_max_size,
                                 &min_reduce, &file_size, min_count);

    int a; int c;

    if (weights == NULL) {
        char tmpstr[100];
        long long tmpint;
        if (debug_mode > 1) printf("Reading word vectors as weights from %s\n", weight_file);
        FILE *fw = fopen(weight_file, "rb");

        if (fw == NULL) {
            if (debug_mode > 0) printf("ERROR: weight file not found!\n");
            exit(1);
        }

        fscanf(fw, "%lld %lld", &tmpint, &layer1_size);
        if (debug_mode > 1) printf("words: %lld, vec size: %lld\n", tmpint, layer1_size);

        npy_intp w_dims = {vocab_size, layer1_size};
        weights = PyArray_SimpleNew(2, w_dims, NPY_FLOAT);
        for (a = 0; a < vocab_size; a++) {
            fscanf(fw, "%s", tmpstr);
            for (c = 0; c < layer1_size; c++) {
                fscanf(fw, "%lf", *((float *)PyArray_GETPTR2(weights, a, c))); // first row contains metadata and first columns contains word
            }
        }
        if (debug_mode > 1) printf("Weights read\n");
    }

    npy_intp *wdims = PyArray_DIMS(weights);
    if (debug_mode > 1) printf("weights' dims %ld %ld\n", wdims[0], wdims[1]);

    PyArrayObject* docvecs = embed_docs(test_file, output_file, layer1_size, sample, train_words, vocab, vocab_hash,
                                        weights);

    for (a = 0; a < vocab_size; a++) {
        free(vocab[a].word);
        free(vocab[a].code);
        free(vocab[a].point);
    }
    free(vocab); free(vocab_hash);
    PyArray_ENABLEFLAGS((PyArrayObject*)docvecs, NPY_ARRAY_OWNDATA);
    return Py_BuildValue("O", docvecs);
}

static PyObject* generate_vocab_file(PyObject *self, PyObject *args, PyObject *kws) {
        struct vocab_word *vocab;
        int debug_mode = 2, min_count = 5, min_reduce = 1;
        int *vocab_hash;
        long long vocab_max_size = 1000, vocab_size = 0;
        long long train_words = 0, file_size = 0;
        char *train_file, *save_vocab_file;
        static char *kwlist[] = {
            "train_file", "save_vocab", "min_count", "debug_mode", NULL
        };
        if (!PyArg_ParseTupleAndKeywords(
          args, kws, "ss|ii", kwlist,
          &train_file,         //s
          &save_vocab_file,    //s
          &min_count,          //i
          &debug_mode          //i
        ))
        if (debug_mode > 1) printf("train file %s, save_vocab_file %s, min_count %d, debug_mode %d\n",
                                    train_file, save_vocab_file, min_count, debug_mode);
         vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
         vocab_hash = (int *)calloc(VOCAB_HASH_SIZE, sizeof(int));

         LearnVocabFromTrainFile(train_file, &train_words, debug_mode, &vocab, vocab_hash, &vocab_size, &vocab_max_size,
                                 &min_reduce, &file_size, min_count);
         SaveVocab(save_vocab_file, vocab, vocab_size);
         if (debug_mode > 0) printf("Vocabulary saved to %s\n", save_vocab_file);
         return Py_BuildValue("");
}

static PyObject* test_method(PyObject *self, PyObject *args) {
    return Py_BuildValue("s", "C module compiled successfully.");
}

static char test_method_docstring[] = "this is a dummy method";

static PyMethodDef module_methods[] = {
    {"train", (PyCFunction)train, METH_VARARGS | METH_KEYWORDS, NULL},
    {"transform", (PyCFunction)transform, METH_VARARGS | METH_KEYWORDS, NULL},
    {"generate_vocab_file", (PyCFunction)generate_vocab_file, METH_VARARGS | METH_KEYWORDS, NULL},
    {"test", (PyCFunction)test_method, METH_NOARGS, test_method_docstring},
    {NULL, NULL, 0, NULL}
};

static char module_docstring[] = "This is doc2vecc for use in python!";

static struct PyModuleDef c_doc2vecc = {
    PyModuleDef_HEAD_INIT,
    "c_doc2vecc",
    module_docstring,
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC PyInit_c_doc2vecc(void) {
    return PyModule_Create(&c_doc2vecc);
}
