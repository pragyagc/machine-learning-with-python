import pandas as pd
music_data = pd.read_csv('E:/python learning/Machine_learning/tutorial1/music.csv')
print(music_data)

 #dropping column
x =music_data.drop(columns=['genre'])
# print(x)
y= music_data['genre']


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x,y)
predictions=model.predict([[21,1],[22,0]])
print(predictions)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
music_data = pd.read_csv('E:/python learning/Machine_learning/tutorial1/music.csv')
x =music_data.drop(columns=['genre'])
y= music_data['genre']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions=model.predict(x_test)

score=accuracy_score(y_test,predictions)
print(score)

import joblib
#create a file
joblib.dump(model,'music-recommender.joblib')

model =joblib.load('music-recommender.joblib')
predictions=model.predict([[21,1]])
print(predictions)


#DECISION TREE
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
music_data = pd.read_csv('E:/python learning/Machine_learning/tutorial1/music.csv')
x =music_data.drop(columns=['genre'])
y= music_data['genre']

model = DecisionTreeClassifier()
model.fit(x,y)
tree.export_graphviz(model,out_file='music-recommender.dot',feature_names=['age','gender'],class_names=sorted(y.unique()),label='all',rounded=True,filled=True)

from graphviz import Source

# Load and render the .dot file
with open("music-recommender.dot", "r") as file:
    dot_code = file.read()

graph = Source(dot_code)
graph.render("music_tree", view=True, format="png")  # Saves and opens music_tree.png

