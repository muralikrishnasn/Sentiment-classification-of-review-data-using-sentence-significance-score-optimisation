import os
import numpy as np
import random

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import classification_report

from sklearn.cluster import k_means
from sklearn.cluster import KMeans

import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import *

from config import Config
from dataGenerator import DataGenerator

class read_file():

	def __init__(self):
		np.random.seed(100)
		self.documents = []
		self.labels = []
		self.files = []
		self.answer = []
		self.num_words = Config.NUM_WORDS
		self.embed = {}
		self.debug = False

	def readFiles(self):

		folder_name = Config.FILE_LOCATION
		print(folder_name)
		for ctr, filename in enumerate(os.listdir(folder_name)):
		    self.files.append(filename)
		    with open(os.path.join(folder_name, filename), encoding='utf-8', errors='ignore') as my_file:
		        for line in my_file:
		            self.documents.append(line)
		print(len(self.documents))

		folder_name = os.getcwd()+'/labelset/'
		

		for filename in os.listdir(folder_name):
		    with open(os.path.join(folder_name, filename), encoding='utf-8', errors='ignore') as my_file:
		        for line in my_file:
		            if(line[0] == 'p'):
		                self.labels.append(1)
		            else:
		                self.labels.append(0)
		print(self.labels[:100], len(self.labels))

		c = list(zip(self.documents, self.labels))
		random.shuffle(c)
		self.documents, self.labels = zip(*c)

	def pre_process_documents(self):

		st = stopwords.words('english')
		st.append('br')
		stop_words = set(st)

		new_documents = []

		for i in range(len(self.documents)):
		#for i in range(100):
			
			word_tokens = word_tokenize(self.documents[i])

			text = ' '.join([word for word in word_tokens if word not in stop_words])
			# print(type(text), type(self.documents[i]))
			new_documents.append(text)
			if i%1000==0:
				print(i)

		self.documents = new_documents
		print(len(self.documents))


	def read_embeddings(self):

		with open(Config.GLOVE_EMBEDDING) as my_file:
			for line in my_file:
				line = line.split()
				self.embed[line[0]] = np.asarray(line[1:], dtype='float32')

	def word_to_index(self):

		text_tokenizer = Tokenizer(lower=True, num_words=self.num_words)
		text_tokenizer.fit_on_texts(self.documents)
		text_sequences = text_tokenizer.texts_to_sequences(self.documents)
		text_index = text_tokenizer.word_index
		#text_index['.'] = 177
		print('Found %s unique tokens.' % len(text_index))
		print(len(text_index.items()))
		inv_map = {v:k for k,v in text_index.items()}
		#print(text_index)
		print(len(text_index))
		return text_tokenizer, text_sequences, text_index, inv_map

	def clustering(self, text_tokenizer, inv_map):

		# for i in len(range(self.documents)):
		encoded_sentence = []
		no_of_sentences = 0
		sentences_in_documents = []
		for i in range(len(self.documents)):
			sentences = sent_tokenize(self.documents[i])
			if self.debug:
				print("Sentences", sentences, len(sentences))
			sentences_in_documents.append(len(sentences))

			sentences = text_tokenizer.texts_to_sequences(sentences)
			if self.debug:
				print("Len of sentences", len(sentences))

			for sentence in sentences:
				result = np.zeros(Config.EMBEDDING_SIZE,dtype='float64')
				for word in sentence:
					if inv_map[word] in self.embed:
						result += self.embed[inv_map[word]]
				if self.debug:
					print(sentence)
				no_of_sentences += 1
				if len(sentence) != 0:
					result /= len(sentence)
				encoded_sentence.append(result)

		
		print(len(encoded_sentence), no_of_sentences)
		print(encoded_sentence[0].shape, encoded_sentence[1].shape)
		encoded_sentence = np.array(encoded_sentence)#np.concatenate(encoded_sentence, axis=0)
		print(encoded_sentence.shape)
		# result = k_means(encoded_sentence, Config.NUMBER_CLUSTERS)
		result = KMeans(n_clusters=Config.NUMBER_CLUSTERS).fit(encoded_sentence)
		# print(len(result[1]))
		print(sum(sentences_in_documents))
		#print(sentences_in_documents)
		# self.embed.clear()

		return result, sentences_in_documents

	def valid_split(self):
		#implement later
		self.train = self.documents[:24000]
		self.train_labels = self.labels[:24000]

		self.valid = self.documents[24000:]
		self.valid_labels = self.labels[24000:]

	def compute(self, text_tokenizer, inv_map, k_means, embed, num_words, comb):
		input_1 = Input(shape=(num_words,))
		#process_1 = Dense(1024)(input_1)
		#process_2 = Activation('tanh')(process_1)
		process_3 = Dense(1)(input_1)
		output_1 = Activation('tanh')(process_3)
		model_1 = Model(input_1, output_1)
		print(model_1.summary())

		input_2 = Input(shape=(4,num_words))
		process_4 = TimeDistributed(model_1)(input_2)
		process_5 = Flatten()(process_4)
		process_6 = Dense(1)(process_5)
		output_2 = Activation('sigmoid')(process_6)
		model_2 = Model(input_2, output_2)
		print(model_2.summary())

		train_generator = DataGenerator(self.train, self.train_labels, text_tokenizer, inv_map, k_means, embed, num_words)
		valid_generator = DataGenerator(self.valid, self.valid_labels, text_tokenizer, inv_map, k_means, embed, num_words)

		model_2.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])
		model_2.fit_generator(generator=train_generator, validation_data=valid_generator, epochs=2, verbose=1)
			
		# a = model_2.predict_generator(data_yield_1(valid_sequences,valid_results,valid_clusters,992),steps=62)
		#print(a[:10])
		#a = np.argmax(a,axis=1)
		#b = np.argmax(valid_results,axis=1)
		#print(a[:100],valid_results[:10],b[:10])
		#print(classification_report(b,a))
		return model_2
	

if __name__ == '__main__':

	readFile = read_file()
	readFile.readFiles()
	readFile.pre_process_documents()
	readFile.read_embeddings()
	text_tokenizer, text_sequences, text_index, inv_map = readFile.word_to_index()
	# print(text_sequences)
	result, _ = readFile.clustering(text_tokenizer, inv_map)
	readFile.valid_split()
	readFile.compute(text_tokenizer, inv_map, result, readFile.embed, Config.NUM_WORDS, 2)