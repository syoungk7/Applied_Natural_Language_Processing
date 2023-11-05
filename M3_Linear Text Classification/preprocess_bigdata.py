from datasets import load_dataset 
from transformers import AutoTokenizer

class Preprocess:
    def __init__(self):
        '''
        Initialize the dataset and the tokenizer. 
        For self.dataset, most of the code is given. We are importing the Wikipedia dataset from Hugging Face.
        However, because of the big data context (size of the dataset is 20GB!), we will need to alter it to 
        allow for streaming (https://huggingface.co/docs/datasets/stream) so that we don't load all of it into
        memory. Note that after allowing for streaming, self.dataset will be of the type IterableDataset, 
        documentation of which can be found here: 
        https://huggingface.co/docs/datasets/v2.14.4/en/package_reference/main_classes#datasets.IterableDataset.
        You could also try running the code once as is to compare streaming vs non-streaming, and see why it is 
        so important!
        For self.tokenizer, let's use the AutoTokenizer we imported from the Hugging Face transformers library. 
        We will use the 'distilbert-base-cased' tokenizer, which is a distilled version of the LLM, BERT. We will
        be learning more about the inner-workings of BERT later on in the course.
        Autotokenizer: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
        Distilbert: https://huggingface.co/distilbert-base-cased
        Tokenizer: https://huggingface.co/docs/transformers/main_classes/tokenizer
        '''
        self.dataset = load_dataset('wikipedia', "20220301.en", split='train', streaming=True)
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

    def tokenize(self, batch, max_length=100):
        '''
        This is a helper method that will be called by preprocess_text() to tokenize the 'text' column in batch. 
        Keep in mind that in this method we are not just tokenizing the text using Distilbert, we are then converting 
        those numerical token ids back to English tokens. Documentation for tokenizers can be found here: 
        https://huggingface.co/docs/transformers/main_classes/tokenizer
        For the tokenizer, we always want the arrays of tokens to be size max_length. Make sure to set the padding and
        truncation strategies to allow for this.
        Args: batch: group of samples streamed from the dataset
              max_length: maximum length of the tokenized text to pad/truncate to
        Return: tokens_dict: dictionary mapping the word 'tokens' to a list of English tokens for each sample in the batch. 
                         This output will then be used in the preprocess_text() method to add a new 'tokens' column to 
                         self.dataset.
        '''
        tokens_dict = dict()
        tmp = []
        for i in batch['text']:
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(i, padding='max_length', truncation=True, max_length=max_length)))
            tmp.append(tokens)
        tokens_dict['tokens'] = tmp
        return tokens_dict

    def preprocess_text(self):
        '''
        Here, we will apply the self.tokenize method on self.dataset. Remember to only keep the columns 'id', 'title', and 'tokens'. Make sure to use batching (use batch of 1000) to allow for proper processing speed. 
        The map function may be useful.
        Return:
            dataset_cleaned: Iterable Dataset with the 'id', 'title', and new 'tokens' column added
        '''
        #print(len(list(self.dataset)))
        #dataset_cleaned = self.tokenize(list(self.dataset), max_length=1000)
        #return dataset_cleaned
