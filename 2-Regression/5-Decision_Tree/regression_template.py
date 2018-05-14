import matplotlib.pyplot as plt

class Regression:
    def __init__(self,X,y, title='Title', xlabel = 'xlabel', ylabel = 'ylabel'):
        self.X = X
        self.y = y
        self.regressor = 0
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot_regression(self,regressor,X_predict):
        plt.scatter(self.X,self.y,color = 'red')
        plt.plot(X_predict,regressor.predict(X_predict),color = 'blue')
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)