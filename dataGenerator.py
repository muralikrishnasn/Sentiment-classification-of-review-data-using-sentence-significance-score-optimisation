import numpy as np
import keras
from nltk.tokenize import sent_tokenize, word_tokenize

from config import Config

#raw document, labels, text_tokenizer, k_means, inv_map
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, documents, labels, text_tokenizer, inv_map, k_means, embed, num_words, batch_size=32):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.documents = documents
        self.indexes = np.arange(len(self.documents))
        self.text_tokenizer = text_tokenizer
        self.k_means = k_means
        self.num_words = num_words
        self.embed = embed
        self.inv_map = inv_map

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.documents) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, document_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        temp_in_0 = [];temp_in_1 = [];temp_in_2 = [];temp_in_3 = []
        temp_out= []

        # Generate data
        for document_no in document_ids:
            sentences = sent_tokenize(self.documents[document_no])
            sentences = self.text_tokenizer.texts_to_sequences(sentences)

            x_temp_0 = np.zeros((1,self.num_words))
            x_temp_1 = np.zeros((1,self.num_words))
            x_temp_2 = np.zeros((1,self.num_words))
            x_temp_3 = np.zeros((1,self.num_words))

            for sentence in sentences:
                result = np.zeros(Config.EMBEDDING_SIZE,dtype='float64')
                for word in sentence:
                    if self.inv_map[word] in self.embed:
                        result += self.embed[self.inv_map[word]]
                # print(sentence)
                # no_of_sentences += 1
                if len(sentence) != 0:
                    result /= len(sentence)
                
                # print('no', type(result), result.shape)
                result = result.reshape(1, -1)
                # print('no', type(result), result.shape)
                cluster_no = self.k_means.predict(result)
                # print('problem')

                temp = self.text_tokenizer.sequences_to_matrix([sentence], mode='binary')

                if cluster_no==0:
                    x_temp_0 +=temp
                elif cluster_no==1:
                    x_temp_1 +=temp
                elif cluster_no==2:
                    x_temp_2 +=temp
                else:
                    x_temp_3 +=temp

            temp_in_0.append(x_temp_0)
            temp_in_1.append(x_temp_1);
            temp_in_2.append(x_temp_2);
            temp_in_3.append(x_temp_3)
        
            temp_out.append(np.reshape(self.labels[document_no], (1,1)))

        temp_in_0 = np.reshape(np.asarray(temp_in_0),(-1,self.num_words))
        temp_in_1 = np.reshape(np.asarray(temp_in_1),(-1,self.num_words))
        temp_in_2 = np.reshape(np.asarray(temp_in_2),(-1,self.num_words))
        temp_in_3 = np.reshape(np.asarray(temp_in_3),(-1,self.num_words))
        temp_out = np.reshape(np.asarray(temp_out),(-1,1))
        #print('\n',times,ctr)
        temp_in = np.dstack((temp_in_0,temp_in_1,temp_in_2,temp_in_3))
        #print(temp_in.shape)
        temp_in = np.swapaxes(temp_in,1,2)
        #print(temp_in.shape,temp_out.shape)
        # print("returning")
        return temp_in,temp_out

