def timefunc(func):
	from time import time
	def f(*args, **kwargs):
		start = time()
		rv = func(*args, **kwargs)
		finish = time()
		print('RUN TIME: ', finish - start)
		return rv
	return f

@timefunc
def count_punct(text):
	import string; import pandas as pd
	count = sum([1 for char in text if char in string.punctuation])
	return round(count/(len(text) - text.count(" ")), 3)*100

def keras_tokenize(data):
	from keras.preprocessing.sequence import pad_sequences
	from keras.preprocessing.text import Tokenizer
	tokenizer = Tokenizer()
	data = data.astype('str')
	tokenizer.fit_on_texts(data)
	print('Embedding(input_dim= 1 + {}'.format(len(tokenizer.word_index)))
	sequences = tokenizer.texts_to_sequences(data)
	x_train_test_padded = pad_sequences(sequences)
	print('input_length={}'.format(len(x_train_test_padded[0])))
	# Insert into model.fit(x, ...)
	return x_train_test_padded

import pandas as pd
import string
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

class SarcTrain(object):
	
	def __init__(self, path):
		self.path = path
		train = pd.read_table(self.path, encoding='latin2', header=None)
		train = train.drop([2, 3, 4, 5, 6, 7, 8, 9], axis=1)
		train.columns = ['sarcasm', 'text']
		train['punct%'] = train['text'].astype('str').apply(lambda x: count_punct(x))
		train['comment_length'] = train['text'].astype('str').apply(lambda x: len(x) - x.count(" "))
		self.train = train

	def X_train(self):
		x_train = keras_tokenize(self.train['text'])
		x_train = pd.concat([self.train[['punct%','comment_length']], pd.DataFrame(x_train)], axis=1)
		print('X_train Embedding -- input_length={}'.format(len(x_train.columns)))
		return x_train

	def y_train(self):
		y_train = self.train['sarcasm'].values
		print(y_train[0:5])
		return y_train

class SarcTest(object):

	def __init__(self, path):
		self.path = path
		test = pd.read_table(self.path, encoding='latin2', header=None)
		test = test.drop([2, 3, 4, 5, 6, 7, 8, 9], axis=1)
		test.columns = ['sarcasm', 'text']
		test['punct%'] = test['text'].astype('str').apply(lambda x: count_punct(x))
		test['comment_length'] = test['text'].astype('str').apply(lambda x: len(x) - x.count(" "))
		self.test = test

	def X_test(self):
		x_test = keras_tokenize(self.test['text'])
		x_test = pd.concat([self.test[['punct%','comment_length']], pd.DataFrame(x_test)], axis=1)
		print(len(x_test.columns))
		return x_test

	def y_test(self):
		y_test = self.test['sarcasm'].values
		print(y_test[0:5])
		return y_test
	
class SarcModels:

	def __init__(self, x, y, in_dim, out_dim, in_len, epochs=1, save_path):
		self.x = x
		self.y = y
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.in_len = in_len
		self.epoch = epoch
		self.save_path = save_path

	def deep_learning_python67(self):
		from keras.models import Sequential
		from keras.layers import Flatten, Dense, Embedding
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.1)
	
		model = Sequential()
		model.add(Embedding(input_dim=self.in_dim, output_dim=self.out_dim, input_length=self.in_len))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))
		model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
		model.summary()
		model.fit(X_train, y_train, epochs=self.epoch, batch_size=32, validation_split=0.2)
		evaluate = model.evaluate(X_test, y_test, verbose=1)
		print('Evaluation -- {}'.format(evaluate))
		model.save(self.save_path + '.h5')
		print('Model Saved to Disk')
	
	def pydata_example(self):
		from keras.models import Sequential
		from keras.layers import Flatten, Dense, Embedding, LSTM, Dropout, Activation
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
	
		model = Sequential()
		model.add(Embedding(input_dim=self.in_dim, output_dim=self.out_dim, input_length=self.in_len))
		model.add(LSTM(units=64))
		model.add(Dropout(0.2))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.summary()
		model.fit(X_train, y_train, epochs=self.epoch, verbose=1)
		evaluate = model.evaluate(X_test, y_test, verbose=1)
		print('Evaluation -- {}'.format(evaluate))
		model.save(self.save_path + '.h5')
		print('Model Saved to Disk')
	
	def load_h5_model(self, model, xtest):
		from keras.models import load_model
		model = load_model(model)
		predict = model.predict(xtest)
		# y_prob = predictions[:,1]
		# loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# score = loaded_model.evaluate(X_test, Y_test, verbose=0)
		# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))














































