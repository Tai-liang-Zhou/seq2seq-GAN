import glob #返回所有匹配的文件路径列表。
import codecs
import json
import data
from random import shuffle #方法将序列的所有元素随机排序。
import tensorflow as tf
import numpy as np
from nltk.tokenize import sent_tokenize

FLAGS = tf.flags.FLAGS



class Example(object):
    def __init__(self, sentence, vocab, hps, input = None):
        
        self.hps = hps
        
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)
        stop_doc = vocab.word2id(data.STOP_DECODING_DOCUMENT)

        if input != None:
            article = input
            article_words = article.split() # list of strings
            if len(article_words) > hps.max_enc_steps:
                article_words = article_words[:hps.max_enc_steps]
            
            self.enc_len = len(article_words)
            self.enc_input = [vocab.word2id(w) for w in article_words]
            self.original_review_input = input
            self.original_review_output = sentence

            # review_sentence = sent_tokenize(sentence) # 按句子做分割
            review_sentence = sentence
            abstract_sentences = [x.strip() for x in review_sentence] # 分割出來的句子
            abstract_words = [] 
            for i in range(len(abstract_sentences)):
                # 避免decoder字數超過最大限制
                if i >= hps.max_dec_sen_num:
                    abstract_words = abstract_words[:hps.max_dec_sen_num]
                    break
                abstract_sen = abstract_sentences[i]
                abstract_sen_words = abstract_sen.split()
                if len(abstract_sen_words) > hps.max_dec_steps:
                    abstract_sen_words = abstract_sen_words[:hps.max_dec_steps]
                abstract_words.append(abstract_sen_words)

            if len(abstract_words[-1]) < hps.max_dec_steps:
                abstract_words[-1].append('[STOP]')
        else:
            # print(sentence)
            review_sentence = sentence
            # review_sentence = sentence.split()
            article = review_sentence[0]
            article_words = article.split()  # list of strings
            if len(article_words) > hps.max_enc_steps:
                article_words = article_words[:hps.max_enc_steps]
            self.enc_len = len(article_words)  # store the length after truncation but before padding
            self.enc_input = [vocab.word2id(w) for w in article_words]  # list of word ids; OOVs are represented by the id for UNK toke

            
            self.original_review_input = review_sentence[0]
            self.original_review_output =  " ".join(review_sentence[1:])
            
            review_sentence = review_sentence[1:]
            abstract_sentences = [x.strip() for x in review_sentence]
            abstract_words = []
            
            for i in range(len(abstract_sentences)):
                if i >= hps.max_dec_sen_num:
                    abstract_words = abstract_words[:hps.max_dec_sen_num]
                    break
                abstract_sen = abstract_sentences[i]
                abstract_sen_words = abstract_sen.split()
                if len(abstract_sen_words) > hps.max_dec_steps:
                    abstract_sen_words = abstract_sen_words[:hps.max_dec_steps]
                abstract_words.append(abstract_sen_words)
            if len(abstract_words[-1]) < hps.max_dec_steps:
                abstract_words[-1].append('[STOP]')
                
        abs_ids = [[vocab.word2id(w) for w in sen] for sen in abstract_words] 
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_sen_num, hps.max_dec_steps,start_decoding,stop_doc)
        self.dec_len = len(self.dec_input)
        self.dec_sen_len = [len(sentence) for sentence in self.target]
        self.original_review = sentence

    def get_dec_inp_targ_seqs(self, sequence, max_sen_num,max_len, start_id, stop_id):
        inps = sequence[:]
        targets = sequence[:]

        if len(inps) > max_sen_num:
            inps = inps[:max_sen_num]
            targets = targets[:max_sen_num]

        for i in range(len(inps)):
            # inps[i] = [start_id] + inps[i][:]
            inps[i] = [start_id]
            if len(inps[i]) > max_len:
                inps[i] = inps[i][:max_len]

        for i in range(len(targets)):
            if len(targets[i]) >= max_len:
                targets[i] = targets[i][:max_len - 1]  # no end_token
                targets[i].append(stop_id)  # end token
            else:
                targets[i]=targets[i] +[stop_id]
        return inps, targets

    def pad_decoder_inp_targ(self, max_sen_len, max_sen_num, pad_doc_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_sen_len) < max_sen_num:
            self.dec_sen_len.append(1)

        for i in range(len(self.dec_input)):
            while len(self.dec_input[i]) < max_sen_len:
                self.dec_input[i].append(pad_doc_id)

        while len(self.dec_input) < max_sen_num:
            self.dec_input.append([pad_doc_id for i in range(max_sen_len)])
        
        for i in range(len(self.target)):
            while len(self.target[i]) < max_sen_len:
                self.target[i].append(pad_doc_id)

        while len(self.target) < max_sen_num:
            self.target.append([pad_doc_id for i in range(max_sen_len)])



    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) <  max_len:
            self.enc_input.append(pad_id)



class GenBatcher(object):
    def __init__(self, vocab, hps):
        self._vocab = vocab
        self._hps = hps
        
        self.train_queue = self.fill_example_queue("review_generation_dataset/train/123.csv", mode ="train")
        self.test_queue = self.fill_example_queue("review_generation_dataset/train/123.csv",  mode ="test")

        self.train_batch = self.create_batch(mode = "train")
        self.test_batch = self.create_batch(mode="test", shuffleis=False)

    def get_batches(self, mode = 'train'):
        if mode == "train":
            shuffle(self.train_batch)
            return self.train_batch
        elif mode == 'test':
            return self.test_batch
        
        

    def fill_example_queue(self, data_path, mode = "test"):
        new_queue = []
        with codecs.open(data_path, 'r', 'utf-8') as ask_f:
            for line in ask_f:
                line = line.split(",")
                for index in range(len(line)):
                    line[index] = line[index].strip()
                    line[index] = line[index].strip('\ufeff')
                example = Example(line,  self._vocab, self._hps)
                new_queue.append(example)
            return new_queue

        # filelist = glob.glob(data_path)
        # assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
        # filelist = sorted(filelist)

        # if mode == "train":
        #     filelist = filelist
        # for f in filelist:
        #     reader = codecs.open(f, 'r', 'utf-8')
        #     while True:
        #         string_ = reader.readline()
        #         if not string_: break
        #         dict_example = json.loads(string_)
        #         review = dict_example["review"]
        #         if(len(sent_tokenize(review)) < 2):
        #             continue
        #         example = Example(review, self._vocab, self._hps)
        #         new_queue.append(example)
        # return new_queue

    def create_batch (self, mode = "train", shuffleis = True):
        all_batch = []

        if mode == "train":
            num_batches = int(len(self.train_queue)/ self._hps.batch_size)
            if shuffleis:
                shuffle(self.train_queue)
        elif mode == "test":
            num_batches = int(len(self.test_queue) / self._hps.batch_size)

        for i in range(0, num_batches):
            batch = []
            
            if mode == "train":
                batch += (self.train_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test':
                batch += (self.test_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            all_batch.append(Batch(batch, self._hps, self._vocab))

        return all_batch
class Batch(object):
    def __init__(self, example_list, hps, vocab):
        self.pad_id = vocab.word2id(data.PAD_TOKEN)
        if FLAGS.run_method == 'auto-encoder':
            self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder
        self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
        self.store_orig_strings(example_list) # store the original strings

    def init_encoder_seq(self, example_list, hps):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            # print (ex.enc_input)
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len

    def init_decoder_seq(self, example_list, hps):

        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, hps.max_dec_sen_num,self.pad_id)
        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_sen_num, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_sen_num, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((hps.batch_size* hps.max_dec_sen_num, hps.max_dec_steps),
                                        dtype=np.float32)
        self.dec_sen_lens = np.zeros((hps.batch_size, hps.max_dec_sen_num), dtype=np.int32)
        self.dec_lens = np.zeros((hps.batch_size), dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.dec_lens[i] = ex.dec_len
            self.dec_batch[i, :, :] = np.array(ex.dec_input)
            self.target_batch[i] = np.array(ex.target)
            for j in range(len(ex.dec_sen_len)):
                self.dec_sen_lens[i][j] = ex.dec_sen_len[j]


        self.target_batch = np.reshape(self.target_batch, [hps.batch_size*hps.max_dec_sen_num, hps.max_dec_steps])

        for j in range(len(self.target_batch)):
            for k in range(len(self.target_batch[j])):
                if int(self.target_batch[j][k]) != self.pad_id:
                    self.dec_padding_mask[j][k] = 1 

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""

        self.original_review_output = [ex.original_review_output for ex in example_list] # list of lists
        if FLAGS.run_method == 'auto-encoder':
            self.original_review_inputs = [ex.original_review_input for ex in example_list]  # list of lists