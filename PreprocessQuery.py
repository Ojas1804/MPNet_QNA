from Lemmatizer import lemmatize_text
from Stopwords import all_stop_words


class PreprocessQuery:
    def __init__(self, query):
        self.query = query
        self.stop_words = all_stop_words()
        self.lemmatize_text = lemmatize_text
        self.preprocessed_query = self.preprocess()


    def get_preprocessed_query(self):
        return self.preprocessed_query
        

    def preprocess(self):
        lemmatized_query = self.lemmatize_text(self.query)
        query = lemmatized_query[0]
        query = [q for q in query if q not in self.stop_words]
        return query