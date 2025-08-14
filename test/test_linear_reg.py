
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.linear_regression import LinearRegression

X = [[1],[2],[3],[4],[5]]
y = [2,4,6,8,10]

model = LinearRegression()
model.fit(X,y)

print("Weights set: ", model.weights)
print("Bias set: ", model.bias)
print("Weights set: ", model.predict([[6]])) # Expecting 12
