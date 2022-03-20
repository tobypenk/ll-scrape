from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast

app = Flask(__name__)
api = Api(app)



class Prediction(Resource):
    
    def __init__(self):
        
        import os
        import pickle
        self.export_path = os.path.join(os.getcwd(), 'pretrained_weights', '20220319')
        
        with open('pretrained_weights/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
            
        with open('pretrained_weights/metaparameters.pickle', 'rb') as handle:
            self.metaparameters = pickle.load(handle)
            
        self.category_map = {
            "1":"AMER HIST",
            "2":"ART",
            "3":"BUS/ECON",
            "4":"CLASS MUSIC",
            "5":"CURR EVENTS",
            "6":"FILM",
            "7":"FOOD/DRINK",
            "8":"GAMES/SPORT",
            "9":"GEOGRAPHY",
            "10":"LANGUAGE",
            "11":"LIFESTYLE",
            "12":"LITERATURE",
            "13":"MATH",
            "14":"POP MUSIC",
            "15":"SCIENCE",
            "16":"TELEVISION",
            "17":"THEATRE",
            "18":"WORLD HIST"
        }
        
    def get(self):
        
        import ast
        parser = reqparse.RequestParser()
        parser.add_argument('questions', required=True)
        args = parser.parse_args()

        self.questions = ast.literal_eval(args['questions'])
        
        return {'data': self.get_answers()}, 200
        
    def get_model(self):
        
        if not hasattr(self, "model"):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.export_path)
            
        return self.model
    
    def predict(self):
        
        if not hasattr(self, "predictions"):
            tmp_model = self.get_model()
            self.predictions = tmp_model.predict(self.get_questions())
        return self.predictions
    
    def get_questions(self):
        
        """if not hasattr(self,"questions"):
            
            
            questions = []
            for s in soup.find_all("div","dont-break-out",limit=6):
                clean_string = self.clean_question_text_string(s.text)
                questions.append(clean_string)"""
        
        from bs4 import BeautifulSoup
        with open('live_page_sample.html', 'r') as f:
                page_data = f.read()
        soup = BeautifulSoup(page_data,'html.parser')
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        questions = list(map(self.clean_question_text_string,self.questions))
        questions = self.tokenizer.texts_to_sequences(questions)
        questions = pad_sequences(
            questions, 
            maxlen = self.metaparameters["max_length"], 
            padding = self.metaparameters["padding_type"], 
            truncating = self.metaparameters["trunc_type"]
        )

        self.questions = questions
        self.soup = soup
            
        return self.questions
        
    def clean_question_text_string(self,q):
        import re
        return re.sub("^[0-9]\.","",q).strip()
        
    def get_opponent_data(self):
        
        if not hasattr(self, "opponent_data"):
            opponent = self.soup.find("div","infobox")
            opponent_data = {}
            for tr in opponent.find_all("tr","datarow"):
                tds = tr.find_all("td")
                category = tds[0].text
                score = tds[2].text
                opponent_data[category] = score
            self.opponent_data = opponent_data
        return self.opponent_data

    def get_answers(self):
        
        if not hasattr(self, "answers"):
            import numpy as np
            answers = []
            counter = 1

            for fc in self.predict():
                this_category = self.category_map[str(np.argmax(fc))]
                opponent_category_performance = self.get_opponent_data()[this_category]
                answers.append([counter,this_category,opponent_category_performance])
                counter += 1

            answers = sorted(answers, key=lambda x: x[2] )
            counter = 1
            for a in answers:
                if counter == 1:
                    a.append(3)
                elif counter < 4:
                    a.append(2)
                elif counter < 6:
                    a.append(1)
                else:
                    a.append(0)
                counter += 1

            self.answers = sorted(answers, key=lambda x: x[0]) 
        return self.answers




api.add_resource(Prediction,'/prediction')



if __name__ == '__main__':
    app.run()


