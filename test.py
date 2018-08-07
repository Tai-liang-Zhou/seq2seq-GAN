import jieba
import codecs
jieba.load_userdict('dir.txt')
with codecs.open('review_generation_dataset/train/new_train_data.csv', 'r', 'utf-8') as ask_f:
    for line in ask_f:
        line = line.split(",")
        sentence = jieba.cut(line[0])
        sentence = (" ".join(sentence))
        sentence2 = jieba.cut(line[1])
        sentence2 = (" ".join(sentence2))
        print (sentence+','+ sentence2)
        wrrit_data = sentence+','+ sentence2
        write_positive_file = codecs.open('review_generation_dataset/train/123.csv', "a", "utf-8")
        write_positive_file.write(wrrit_data)
#        print(line)

with codecs.open('review_generation_dataset/train/123.csv', 'r', 'utf-8') as ask_f:
    a = []
    for line in ask_f:
        line = line.split(",")
        for index in range(len(line)):
                    line[index] = line[index].strip()
                    line[index] = line[index].strip('\ufeff')
        a.append(line)
        print(line)


