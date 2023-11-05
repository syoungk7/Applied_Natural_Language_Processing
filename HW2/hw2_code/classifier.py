import dask.dataframe as dd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

class Classifier(object):
    def __init__(self):
        pass

    def read_dask_dataframe(self, filePath):
        """
        Read the csv file referenced in filePath into a Dask Dataframe.
        Name the columns as 'sentiment', 'title', 'text'.
        Args: filePath: str, path of the csv file to load
        Return: df: dask dataframe of csv file with columns 'sentiment', 'title', 'text' 
        """
        return dd.read_csv(filePath, header=None, names=['sentiment', 'title', 'text'])

    def combine_text(self, df):
        """
        Concatenate the 'title' field and 'text' field to create a 'content' field. Use a ' ' to join the two items.
        Args: df: dask dataframe
        Return: df: dask dataframe with added 'content' field
        """
        return df.assign(content=lambda x: x['title'] + ' ' + x['text'])

    def preprocess_text(self, text):
        """
        Convert text to lowercase and remove punctuation, numbers, and line breaks
        NOTE: This function is given and does not have to be implemented or changed.
        Args: text: str, text to be preprocessed
        Return: cleaned_text: str, preprocessed text
        """
        return re.sub(r'[^\w\s]|\d|\r|\n', '', str(text)).lower()

    def compute_preprocess_parallel(self, df):
        """
        Use Dask Dataframe's map function to apply preprocess_text to the 'content' field of the dask dataframe
        and store it in dataframe 'text_processed' column. Then convert it to pandas dataframe and return.
        Args: df: dask dataframe
        Return: df: pandas dataframe with preprocessed 'content' in a new 'text_processed' field
        Hint: Use Dask's map function to apply the preprocess function to Dask dataframe.
        And use Dask's compute function to lazy compute and get a Pandas Dataframe
        If your code errors using map, please confirm that you have dask version 2023.5.0 or greater
        """
        df = df.assign(text_processed = df.content.map(self.preprocess_text))
        return df.compute()

    def create_features_and_labels(self, df):
        """
        Use the data to create and separate features and labels.
        Dataframe 'text_processed' column is used for features and 'sentiment' column is used for labels.
        Encode the features using sklearn's CountVectorize with english stop words and return as x.
        For labels, replace any sentiment label 1 with 0 and sentiment label 2 with 1 in place and return as y.
        Args: df: pandas dataframe
        Return: x: (N, D) csr sparse matrix, vectorized representation of 'text_processed' field
                y: (N,) pd series of adjusted sentiment labels
        Hint: Initialize CountVectorizer with "english" stop words and use fit_transform for updating features.
        Use dataframe replace with inplace=True for updating labels.
        """
        x, y = df["text_processed"], df["sentiment"]
        cv = CountVectorizer(stop_words='english')
        csm = cv.fit_transform(x)
        y.replace({1:0, 2:1}, inplace=True)
        return csm, y

    def train_test_split_fn(self, x, y):
        """
        Function to split train and test data
        NOTE: This function is given and does not have to be implemented or changed.
        Args: x: (N, D) csr sparse matrix
              y: (N,) pd series
        Return: x_train: sparse matrix
                x_test: sparse matrix
                y_train: pd series
                y_test: pd series
        """
        x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=.1)
        return x_train,x_test,y_train,y_test

    def init_XGBoost(self):
        """
        Initialize and return XGBClassifier instance with parallel threads using n_jobs.
        The n_jobs field needs to be set to a value greater than 1.
        Return: clf: XGBClassifier with parallel threads
        """
        return XGBClassifier(n_jobs=3)

    def train_XGBoost(self, clf, x_train, y_train):
        """
        Trains the passed in classifier
        NOTE: This function is given and does not have to be implemented or changed.
        Args: clf: XGBClassifier
              x_train: (N, D) csr sparse matrix
              y_train: (N,) pd series
        Return: clf: trained XGBClassifier
        """
        return clf.fit(x_train, y_train)

    def predict_XGBoost(self, clf, x_test):
        """
        Predict function
        NOTE: This function is given and does not have to be implemented or changed.
        Args: clf: trained XGBClassifier
              x_test: (N, D) csr sparse matrix
        Return: pred: (N,) numpy vector
        """
        return clf.predict(x_test)
