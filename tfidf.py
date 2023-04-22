# ddd.qqq = lnc.ltn
import math
from PostingList import PostingList
import os
from collections import OrderedDict


class tfidf:
    def __init__(self):
        pl = PostingList()
        self.posting_list = pl.get_posting_list()
        # self.N = len(os.listdir("Dataset"))
        self.N=1000
        self.tfidf_wts = {}


    def log_tf(self, tf):
        if tf == 1:
            return 1
        return 1 + math.log10(tf)
    

    def idf(self, df, isQuery):
        if not isQuery:
            return 1
        if df == 0:
            return 0
        return math.log10(self.N/df)
    
    
    def normalize(self, dic, isQuery):
        if isQuery:
            return dic
        sum = 0
        for key in dic:
            sum += dic[key] ** 2
        sum = math.sqrt(sum)
        for key in dic:
            dic[key] /= sum
        return dic
    
    
    def query_tfidf(self, query):
        dic = {}
        for word in query:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
        for key in dic:
            if key not in self.posting_list:
                dic[key] = 0
                continue
            idf = self.idf(self.posting_list[key][0], isQuery=True)
            dic[key] = self.log_tf(dic[key]) * idf
        dic = self.normalize(dic, isQuery=True)
        return dic
    
    
    def tf_idf(self, is_query):
        for key in self.posting_list:
            idf = self.idf(self.posting_list[key][0], is_query)
            dic = self.posting_list[key][1]
            for key2 in dic:
                dic[key2] = self.log_tf(dic[key2]) * idf
            self.posting_list[key][1] = self.normalize(dic, is_query)
        self.tfidf_wts = self.posting_list
        self.store_tfidf_weights("tfidf_weights/weights.txt")


    def get_tfidf_weights(self):
        exist = os.path.exists("tfidf_weights/weights.txt")
        if exist:
            if os.stat("tfidf_weights/weights.txt").st_size != 0: # check if file is not empty
                file = open('tfidf_weights/weights.txt',mode='r')
                text = str(file.read())
                file.close()
                dictionary = eval(text)
                self.tfidf_wts = dictionary
        else:
            self.tf_idf(is_query=False)


    # saves posting lists to a file
    def store_tfidf_weights(self, filename):
        f = open(filename, 'w')
        f.write(str(self.tfidf_wts))
        f.close()

    def get_N(self):
        return self.N


if __name__ == '__main__':
    tfidf = tfidf()
    tfidf.get_tfidf_weights()
    print(tfidf.tfidf_wts)