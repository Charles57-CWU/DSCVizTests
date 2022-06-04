import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin/'
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import graphviz
import numpy as np

df = pd.read_csv('Maverick_Tests/LUPI_BCI_ALL.csv')
#scaler = MinMaxScaler((0, 1))
#df[:] = scaler.fit_transform(df[:])

df_mod = df.copy()
targets = df_mod['class'].unique()
lenn = len(df_mod.columns)
map_to_int = {name: n for n, name in enumerate(targets)}
df_mod['target'] = df_mod['class'].replace(map_to_int)
df_mod = df_mod.drop(['class'], axis=1)
features = list(df_mod.columns[:lenn-1])

print(df_mod)
print(targets)

print(features)
Y = df_mod['target']
X = df_mod[features]
print(np.shape(X))

clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X, Y)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render()
