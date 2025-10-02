# from train import train_model
import pickle
import os
from newspaper import Article
from preprocess_data import Preprocess
import tensorflow
from tensorflow.keras.models import load_model
from train import train_model


class User:
    def __init__(self):
       print("Constructor called")
       self.pre=Preprocess()
       with open("model/vectorizer.pkl", "rb") as f:
         self.vect=pickle.load(f)
       mod_path="model/Fake_news_mod.keras"
       if os.path.exists(mod_path):
          print("Model loaded")
          self.model=load_model(mod_path)
       else:
          print("Model not found")
          self.model=train_model()

    def predictions(self,data):
       percent = self.model.predict(data)[0][0]  # get the scalar probability
       if percent >= 0.5:
          return f"The news likely appears to be True with {percent*100:.2f}% accuracy"
       else:
          return f"The news likely appears to be Fake with {(1-percent)*100:.2f}% accuracy"
    
    def news(self, link: str):
     article = Article(link)
     article.download()
     article.parse()
     data = article.title + " " + article.text
     cleaned = self.pre.clean_text(data)
     # vectorize (wrap in list)
     X = self.vect.transform([cleaned])
     return self.predictions(X)

    
    
    

if __name__=="__main__":
   u1=User()
   link=input("Enter the link here: ")
   print(u1.news(link))    
   
       
        
    

    
