import tensorflow as tf
import time
import data
import json
import codecs
import os
import re
import shutil
import nltk
from  result_evaluate import Evaluate

FLAGS = tf.app.flags.FLAGS

class Generated_sample(object):
    def __init__(self, model, vocab, batcher, sess):
        self._model = model
        self._vocab = vocab
        self._sess = sess

        self.batches = batcher.get_batches(mode = 'train')
        self.test_batches = batcher.get_batches(mode='test')

        self.current_batch = 0

        if not os.path.exists("discriminator_train"): os.mkdir("discriminator_train")
        if not os.path.exists("discriminator_test"): os.mkdir("discriminator_test")
        self.train_sample_whole_positive_dir = os.path.join("discriminator_train","positive")
        self.train_sample_whole_negative_dir = os.path.join("discriminator_train","negative")
        self.test_sample_whole_positive_dir = os.path.join("discriminator_test", "positive")
        self.test_sample_whole_negative_dir = os.path.join("discriminator_test", "negative")
        if not os.path.exists(self.train_sample_whole_positive_dir): os.mkdir(self.train_sample_whole_positive_dir)
        if not os.path.exists(self.train_sample_whole_negative_dir): os.mkdir(self.train_sample_whole_negative_dir)
        if not os.path.exists(self.test_sample_whole_positive_dir): os.mkdir(self.test_sample_whole_positive_dir)
        if not os.path.exists(self.test_sample_whole_negative_dir): os.mkdir(self.test_sample_whole_negative_dir)
        self.temp_positive_dir = ""
        self.temp_negative_dir =""

    def generator_sample_example(self, positive_dir, negative_dir, num_batch):
        self.temp_positive_dir = positive_dir
        self.temp_negative_dir = negative_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        shutil.rmtree(self.temp_negative_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        counter = 0

        for i in range(num_batch):
            decode_result = self._model.run_eval_given_step(self._sess, self.batches[self.current_batch])
            for i in range(FLAGS.batch_size):
                decoded_words_all = []
                original_review = self.batches[self.current_batch].original_review_output[i]
                for j in range(FLAGS.max_dec_sen_num):
                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words
                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)

                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
                self.write_negtive_temp_to_json(original_review, decoded_words_all, counter)
                counter += 1  # this is how many examples we've decoded
            self.current_batch +=1
            if self.current_batch >= len(self.batches):
                self.current_batch = 0
        
        eva = Evaluate()
        eva.diversity_evaluate(negative_dir + "/*")
    def generator_test_sample_example(self, positive_dir, negative_dir, num_batch):
        self.temp_positive_dir = positive_dir
        self.temp_negative_dir = negative_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        shutil.rmtree(self.temp_negative_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        
        counter = 0
        batches = self.test_batches
        step = 0
        list_hop = []
        list_ref = []
        while step < num_batch:
            batch = batches[step]
            step += 1
            decode_result = self._model.run_eval_given_step(self._sess, batch)


            for i in range(FLAGS.batch_size):

                decoded_words_all = []
                original_review = batch.original_review_output[i]

                for j in range(FLAGS.max_dec_sen_num):

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
                self.write_negtive_temp_to_json(original_review, decoded_words_all, counter)
                list_ref.append([nltk.word_tokenize(original_review)])
                list_hop.append(nltk.word_tokenize(decoded_words_all))

                counter += 1  # this is how many examples we've decoded
        
        # bleu_score = corpus_bleu(list_ref, list_hop)
        # tf.logging.info('bleu: '  + str(bleu_score))
        eva = Evaluate()
        eva.diversity_evaluate(negative_dir + "/*")

    def generator_test_max_example(self, positive_dir, negative_dir, num_batch):
        self.temp_positive_dir = positive_dir
        self.temp_negative_dir = negative_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        shutil.rmtree(self.temp_negative_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        
        counter = 0
        batches = self.test_batches
        step = 0
        list_hop = []
        list_ref = []
        while step < num_batch:
            batch = batches[step]
            step += 1
            decode_result = self._model.max_generator(self._sess, batch)

            for i in range(FLAGS.batch_size):

                decoded_words_all = []
                original_review = batch.original_review_output[i]

                for j in range(FLAGS.max_dec_sen_num):

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words) < 2:
                        continue

                    if len(decoded_words_all) > 0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all) - 1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
                self.write_negtive_temp_to_json(original_review, decoded_words_all, counter)
                list_ref.append([nltk.word_tokenize(original_review)])
                list_hop.append(nltk.word_tokenize(decoded_words_all))

                counter += 1  # this is how many examples we've decoded
        
        # bleu_score = corpus_bleu(list_ref, list_hop)
        # tf.logging.info('bleu: '  + str(bleu_score))
        eva = Evaluate()
        eva.diversity_evaluate(negative_dir + "/*")

    def generator_train_negative_example(self):
    
        counter = 0
        step = 0
        
        batches = self.batches
        print(len(batches))
        while step < len(batches):
            
            batch = batches[step]
            step += 1

            decode_result = self._model.run_eval_given_step(self._sess, batch)

            for i in range(FLAGS.batch_size):
                decoded_words_all = []
                original_review = batch.original_review_output[i]  # string

                for j in range(FLAGS.max_dec_sen_num):

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    # print(output_ids)

                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # print("decoded_words :",decoded_words)
                    if decoded_words[0] == '[STOPDOC]':
                        decoded_words = decoded_words[1:]
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
                        decoded_words.append('.')
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)

                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", "", decoded_words_all)

                self.write_negtive_to_json(batch.original_review_inputs[i], original_review, decoded_words_all, counter, self.train_sample_whole_positive_dir, self.train_sample_whole_negative_dir)

                counter += 1  # this is how many examples we've decoded

    def generator_test_negative_example(self):
    
        counter = 0
        step = 0

        # t0 = time.time()
        batches = self.test_batches
        print(len(batches))
        while step < len(batches):
            
            batch = batches[step]
            step += 1

            decode_result =self._model.run_eval_given_step(self._sess, batch)

            for i in range(FLAGS.batch_size):
                decoded_words_all = []
                original_review = batch.original_review_output[i]  # string
                for j in range(FLAGS.max_dec_sen_num):
                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # print("decoded_words :",decoded_words)
                    if decoded_words[0] == '[STOPDOC]':
                        decoded_words = decoded_words[1:]
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words) < 2:
                        continue
                    if len(decoded_words_all) > 0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all) - 1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
                        decoded_words.append('.')
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)

                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                    
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", "", decoded_words_all)

                self.write_negtive_to_json(batch.original_review_inputs[i], original_review, decoded_words_all, counter, self.test_sample_whole_positive_dir,self.test_sample_whole_negative_dir)

                counter += 1  # this is how many examples we've decoded
        # eva = Evaluate()
        # eva.diversity_evaluate(negative_dir + "/*")
        # bleu_score = corpus_bleu(list_ref, list_hop)
        # tf.logging.info('bleu: ' + str(bleu_score))
        

    def write_negtive_temp_to_json(self, positive, negative, counter):
        positive_file = os.path.join(self.temp_positive_dir, "%06d.txt" % ((counter // 1000)))
        negative_file = os.path.join(self.temp_negative_dir, "%06d.txt" % ((counter // 1000)))
        write_positive_file = codecs.open(positive_file, "a", "utf-8")
        write_negative_file = codecs.open(negative_file, "a", "utf-8")

        dict = {"example": str(positive),"label": str(1)}
        string_ = json.dumps(dict, ensure_ascii=False)
        write_positive_file.write(string_ + "\n")

        dict = {"example": str(negative),"label": str(0)}
        string_ = json.dumps(dict, ensure_ascii=False)
        write_negative_file.write(string_ + "\n")

        write_negative_file.close()
        write_positive_file.close()

    def write_negtive_to_json(self, inputs, positive, negative, counter, positive_dir, negtive_dir):
        positive_file = os.path.join(positive_dir, "%06d.txt" % (counter // 1000))
        negative_file = os.path.join(negtive_dir, "%06d.txt" % (counter // 1000))
        write_positive_file = codecs.open(positive_file, "a", "utf-8")
        write_negative_file = codecs.open(negative_file, "a", "utf-8")
        dict = {"inputs":str(inputs),"example": str(positive),"label": str(1)}
        string_ = json.dumps(dict, ensure_ascii=False)
        write_positive_file.write(string_ + "\n")

        dict = {"example": str(negative),"label": str(0)}
        string_ = json.dumps(dict, ensure_ascii=False)

        write_negative_file.write(string_ + "\n")
        
        write_negative_file.close()
        write_positive_file.close()