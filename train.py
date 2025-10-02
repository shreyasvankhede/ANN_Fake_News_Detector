from preprocess_data import Preprocess
import pickle
from sklearn.model_selection import train_test_split
import tensorflow
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model():
    pre=Preprocess()
    df=pre.mergedata()
    df['processed']=df['Full_Text'].apply(pre.clean_text)
    X=df['processed']
    Y=df['label']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    vectorizer=TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train) #doing this after splitting to prevent memory leakage
    X_test=vectorizer.transform(X_test)
    with open('model/vectorizer.pkl', 'wb') as f:
     pickle.dump(vectorizer, f)
    model=Sequential([
     Dense(128,activation='relu',input_shape=(X_train.shape[1],)),
     Dense(64,activation='relu'),
     Dense(1,activation='sigmoid') #For binary classification
     ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train,Y_train,epochs=5,batch_size=32,validation_split=0.1)
    os.makedirs('model', exist_ok=True)
    model.save('model/Fake_news_mod.keras')
    return model

