import codecs

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'
STOP_DECODING_DOCUMENT = '[STOPDOC]'

class Vocab(object):

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # 計算所有在vocab裡面的字

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING, STOP_DECODING_DOCUMENT]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # 讀檔然後增加字數
        with codecs.open(vocab_file, 'r', 'utf-8') as vocab_f:
            for line in vocab_f:
                line = line.strip()
                self._word_to_id[line] = self._count
                self._id_to_word[self._count] = line
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        print("Finished constructing vocabulary of %i total words." % (self._count))
    
    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            return self._id_to_word[0]
        return self._id_to_word[word_id]
        
    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

def outputids2words(id_list, vocab, article_oovs):
    """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

    Args:
        id_list: list of ids (integers)
        vocab: Vocabulary object
        article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

    Returns:
        words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words