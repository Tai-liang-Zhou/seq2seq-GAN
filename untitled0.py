ge_model = Generator(hps_generator, vocab)
sess_ge, saver_ge, train_dir_ge = setup_training_generator(ge_model)

for epoch in range(5000):
    step = 0
    batchers = batcher.get_batches(mode = 'train')
    while step < len(batchers):
        current_batch = batchers[step]
        results = ge_model.run_pre_train_step(sess_ge, current_batch)
        print('step : {}， global_step :{}，loss : {}'.format(step, results['global_step'], results['loss']))
        step+=1
    epoch+=1
    print('trained epoch : ',epoch)
    decode_result = ge_model.run_eval_given_step(sess_ge, current_batch)
    output_ids = [int(t) for t in decode_result['generated'][0][0]][1:]
    decoded_words = data.outputids2words(output_ids, vocab, None)
    print("decoded_words :",decoded_words)
    

decode_result = ge_model.run_eval_given_step(sess_ge, current_batch)
a = decode_result['generated']
output_ids = [int(t) for t in decode_result['generated'][0][0]][1:]
decoded_words = data.outputids2words(output_ids, vocab, None)
print("decoded_words :",decoded_words)



decode_result = ge_model.sample_generator(sess_ge, current_batch)
a = decode_result['generated']
output_ids = [int(t) for t in decode_result['generated'][0][0]][1:]
decoded_words = data.outputids2words(output_ids, vocab, None)
print("decoded_words :",decoded_words)

decode_result = ge_model.max_generator(sess_ge, current_batch)
a = decode_result['generated']
output_ids = [int(t) for t in decode_result['generated'][0][0]][1:]
decoded_words = data.outputids2words(output_ids, vocab, None)
print("decoded_words :",decoded_words)


for i in range(32):
    print(current_batch.original_review_inputs[i])
#    print(current_batch.original_review_output[i])
    output_ids = [int(t) for t in decode_result['generated'][i][0]][1:]
    decoded_words = data.outputids2words(output_ids, vocab, None)
    
    try:
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
    
    print("decoded_words :",decoded_words_all)
    
    
output_ids = [int(t) for t in decode_result['generated'][11][0]][1:]
decoded_words = data.outputids2words(output_ids, vocab, None)
print("decoded_words :",decoded_words)





