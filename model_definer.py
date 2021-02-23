from tensorflow.keras import layers, models, optimizers


# define cnn model
def define_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2, 2)))
    # increase depth of feature extractor part, to improve model accuracy
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='softmax'))
    # compile model, making use of the stochastic gradient descent optimizer,
    # and categorical cross entropy to calculate loss
    opt = optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
