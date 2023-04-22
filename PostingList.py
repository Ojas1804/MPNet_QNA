from Lemmatizer import lemmatize_text
import os
from Stopwords import all_stop_words
from collections import OrderedDict
import pandas as pd


class PostingList:
    def __init__(self):
        self.index = {}
        self.file_index = {}
        self.stop_words = all_stop_words()
        self.num_of_files = 0


    def file_to_posting_list(self, data_instance, file_index):
        # read file and get the text
        # f = open(filename, 'r')
        text = f"{data_instance[0]} {data_instance[1]}"
        # text = f.read()
        text = text.lower()
        # f.close()

        # lemmatize and tokenize text
        tokenized_sentences = lemmatize_text(text)

        # creating the index
        for tokens in tokenized_sentences:
            for token in tokens:
                if token not in self.stop_words:
                    if token in self.index:
                        if file_index in self.index[token][1]:
                            self.index[token][1][file_index]+=1
                        else:
                            self.index[token][0] += 1
                            self.index[token][1][file_index] = 1
                    else:
                        self.index[token] = [1, {}]
                        self.index[token][1][file_index] = 1   # term frequency


    # creating inverse index for each file in dataset
    def data_to_posting_list(self, dirname='Dataset'):
        i = int(0)
        df = pd.read_csv(f"{dirname}/question_answer.csv")
        df.drop(['id'], axis = 1, inplace = True) 
        data = list(df.values)
        # for filename in os.listdir(dirname):
        for i in range(0, len(data)):
            self.file_index[i] = data[i][0]
            self.file_to_posting_list(data[i], i)
            i += 1
        self.num_of_files = i


    # creates posting list
    def create_posting_list(self, filename='Indexes/posting_list.txt'):
        self.data_to_posting_list()
        self.index = OrderedDict(sorted(self.index.items()))
        # print(self.index)
        self.store_file_index("Indexes/file_index.txt")
        self.store_posting_list(filename)


    def get_file_index(self):
        exist = os.path.exists("Indexes/file_index.txt")
        if exist:
            if os.stat("Indexes/file_index.txt").st_size != 0: # check if file is not empty
                file = open('Indexes/file_index.txt',mode='r')
                text = str(file.read())
                file.close()
                dictionary = eval(text)
                self.file_index = dictionary
                return self.file_index
        else:
            self.create_posting_list()
            return self.file_index
        

    def get_posting_list(self):
        exist = os.path.exists("Indexes/posting_list.txt")
        # print(exist)
        if exist:
            if os.stat("Indexes/posting_list.txt").st_size != 0: # check if file is not empty
                file = open('Indexes/posting_list.txt',mode='r')
                text = str(file.read())
                file.close()
                dictionary = eval(text)
                self.index = dictionary
                return self.index
        else:
            self.create_posting_list()
            return self.index
        

    # saves posting lists to a file
    def store_posting_list(self, filename):
        f = open(filename, 'w')
        f.write(str(self.index))
        f.close()


    def store_file_index(self, filename):
        f = open(filename, 'w')
        f.write(str(self.file_index))
        f.close()


if __name__ == '__main__':
    p = PostingList()
    p.get_posting_list()
    print(p)