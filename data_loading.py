from keras.datasets import mnist
from keras.utils import to_categorical


# Load train and test mnist datasets
def load_dataset():
    # Load dataset
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # Reshape dataset to have a single color channel
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    # One hot encode target values
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_x, train_y, test_x, test_y


