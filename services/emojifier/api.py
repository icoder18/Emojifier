import json
from keras.models import model_from_json
import pandas as pd
import numpy as np
import emoji

emoji_dictionary = { '0':'\u2764\uFE0F',
                    '1':':baseball:',
                    '2':':grinning_face_with_big_eyes:',
                    '3':':disappointed_face:',
                    '4':':fork_and_knife:',
                    '5':':hundred_points:',
                    '6':':fire:',
                    '7':':face_blowing_a_kiss:',
                    '8':':chestnut:',
                    '9':':flexed_biceps:'
                    }

		
with open('services/emojifier/models.json','r') as f:
	model=model_from_json(f.read())
model.load_weights('services/emojifier/model.h5')

with open('services/emojifier/glove.6B.50d.txt',encoding='utf-8') as f:
		lines = f.readlines()

embeddings_idx = {}

for l in lines:
	values = l.split()
	word = values[0]
	coeff = np.asarray(values[1:],dtype='float')
	embeddings_idx[word]=coeff

def embedding_output(X):
    maxLen = 10
    embedding_out = np.zeros((X.shape[0],maxLen,50))
    for ix in range(X.shape[0]):
        X[ix]=X[ix].split()
        for ij in range(len(X[ix])):
            try:
                embedding_out[ix][ij]=embeddings_idx[X[ix][ij].lower()]
            except:
                embedding_out[ix][ij]=np.zeros((50,))
    
    return embedding_out              


def predict(x):
	X = pd.Series([x])
	emb_X = embedding_output(X)
	p = model.predict_classes(emb_X)
	return emoji.emojize(emoji_dictionary[str(p[0])])


if __name__=='__main__':
	print(predict("Hello how are you"))