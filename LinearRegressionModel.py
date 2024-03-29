import torch
import matplotlib.pyplot as plt
from torch import nn

from pathlib import Path

# Create and train a Linear Regression model which fits
# Linear function with weight = 0.7 and bias = 0.3

########################## Model ##########################

# Create Linear Regression model by subclassing nn.Module
class LinearRegressionModel(nn.Module):
    
    def __init__(self):
        super().__init__()
    
        # Linear Regression model parameters using nn.Linear (applies a linear transformation to incoming data)
        self.linear_layer = nn.Linear(in_features = 1, out_features = 1)
    
    # define what happens during the forward pass
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


########################## Helper functions ##########################


def visualize_data(train_data, train_labels, test_data, test_labels, predictions):
    
    plt.figure(figsize = (10, 7)) 
    
    # plot training data in blue
    plt.scatter(train_data, train_labels, c = "b", s = 4, label = "Training data")

    # plot testing data in green 
    plt.scatter(test_data, test_labels, c = "g", s = 4, label = "Testing data")

    # plot testing data in red 
    plt.scatter(test_data, predictions, c = "r", s = 4, label = "Predicted data")
 
    plt.show()


def create_linear_data():
    
    # input
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim = 1)
    
    # labels 
    weight = 0.7
    bias = 0.3
    y = weight * X + bias

    # split into training data (80%) and testing data (20%)
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    
    return X_train, y_train, X_test, y_test

def train_model(model, train_data, train_labels):
    
   
   
    # use Mean Absoulute Error as loss function
    loss_function = nn.L1Loss()
    
    # use Stochastic Gradient Descent as optimizer, learning rate = 0.01
    optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.01)
    
    epochs = 200

    print(f"Training the Linear Regression model to fit Linear function (weight = 0.7, bias = 0.3)")
    print(f"Model state before training {model_1.state_dict()}")
    print("###################################################")

    for epoch in range(epochs):
        
        # set model to training mode
        model.train()
    
        # forward pass
        y_pred = model(train_data)

        # calculate the loss
        loss = loss_function(y_pred, train_labels)

        # reset the gradients
        optimizer.zero_grad()

        # back propogation
        # computes the gradient of loss wrt to all parameters in the model,
        # store the gradient in parameter.grad attribute for all the parameters
        loss.backward()

        # updates model parameters based on parameter.grad 
        optimizer.step()

        # test if SGD is making things better
        # set model to evaluation mode
        model.eval()

        with torch.inference_mode():
            test_pred = model(train_data)
            test_loss = loss_function(test_pred, train_labels)

            if epoch % 10 == 0:
                print(f"epoch : {epoch} loss : {loss} test_loss : {test_loss}")
    
    print("###################################################")
    print(f"Model state after training {model_1.state_dict()}")


def evaluate_model(model, test_data):

    model.eval()
    with torch.inference_mode():
        model_pred = model(test_data)
        return model_pred


########################## Test  ##########################

# create test and train data
train_data, train_labels, test_data, test_labels = create_linear_data()

# visualize data
# visualize_data(train_data, train_labels, test_data, test_labels, None)

# set manual seed for reproducibility 
torch.manual_seed(42)
 
# create model instance
model_1 = LinearRegressionModel()

# train model
train_model(model_1, train_data, train_labels)

# save trained model to file
model_save_dir = "models"
Path(model_save_dir).mkdir(parents = True, exist_ok = True)  

model_filename = "model_1.pth"
model_save_path = rf"{model_save_dir}/{model_filename}" 
torch.save(obj = model_1.state_dict(), f = model_save_path)

# load model from file
loaded_model_1 = LinearRegressionModel()
loaded_model_1.load_state_dict(torch.load(model_save_path))

# evaluate model
model_pred = evaluate_model(loaded_model_1, test_data)

# visualize test data, train data and model predictions
visualize_data(train_data, train_labels, test_data, test_labels, model_pred)

