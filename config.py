import os

class Config():
	

		FILE_LOCATION=os.getcwd()+'/dataset/'
		NUM_WORDS=20000
		GLOVE_EMBEDDING='glove.6B.50d.txt'
		EMBEDDING_SIZE = 50
		NUMBER_CLUSTERS = 4