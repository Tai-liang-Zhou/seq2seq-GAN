# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:37:34 2018

@author: tom
"""

import jieba


ge_model = Generator(hps_generator, vocab)
sess_ge, saver_ge, train_dir_ge = setup_training_generator(ge_model)
util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
print("finish load train-generator")




jieba.load_userdict('dir.txt')
inputs = input("Enter your ask: ")
sentence = jieba.cut(inputs)
sentence = (" ".join(sentence))
sentence = sentence.split( )
enc_input = [vocab.word2id(w) for w in sentence]
enc_lens = np.array([len(enc_input)])
enc_input = np.array([enc_input])


out_sentence = ('').split( )
dec_batch = [vocab.word2id(w) for w in out_sentence]
dec_batch = [2] + dec_batch
dec_batch.append(3)
while len(dec_batch) < 40:
    dec_batch.append(1)
    
dec_batch = np.array([dec_batch]).shape
dec_batch = np.resize(dec_batch,(1,1,40))
#dec_lens = np.resize(dec_lens,(1,1,40))
dec_lens = np.array([len(dec_batch)])

result = ge_model.run_test_language_model(sess_ge, enc_input, enc_lens, dec_batch , dec_lens)


output_ids = [int(t) for t in result['generated'][0][0]][1:]
decoded_words = data.outputids2words(output_ids, vocab, None)
print("decoded_words :",decoded_words)
try:
    if decoded_words[0] == '[STOPDOC]':
        decoded_words = decoded_words[1:]
    fst_stop_idx = decoded_words.index(data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
    decoded_words = decoded_words[:fst_stop_idx]
except ValueError:
    decoded_words = decoded_words

if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
    decoded_words.append('.')
decoded_words_all = []
decoded_output = ' '.join(decoded_words).strip()  # single string
decoded_words_all.append(decoded_output)
decoded_words_all = ' '.join(decoded_words_all).strip()
decoded_words_all = decoded_words_all.replace("[UNK] ", "")
decoded_words_all = decoded_words_all.replace("[UNK]", "")
decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
decoded_words_all,_ = re.subn(r"(\. ){2,}", "", decoded_words_all)
if decoded_words_all.startswith('，'):
    decoded_words_all = decoded_words_all[1:]


print("the resonse :",decoded_words_all)
