# Cat-and-Dog-Image-Classification

![cat-and-dog-1](https://github.com/radiadus/Cat-and-Dog-Image-Classification/assets/55176713/35bdcf39-76b3-4ec9-b118-32abfd3e9741)

This project will do image classification for 25.000 images of cats and dogs. You can get the images from kaggle in this website: https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset.

First lets import everything we need for this classification.

    # Import libraries
    import cv2 as cv
    import os
    import io
    from matplotlib import pyplot as plt
    import numpy as np

Then read our images, turn them into gray and append them into lists. Make sure you change the lists into numpy array before go to the next step.

    # Split training dan testing dataset
    from sklearn.model_selection import train_test_split
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

Import train_test_split from sklearn and use it to split our data into training dataset and testing dataset.

    # Reshape features
    x_train = x_train.reshape(-1, 100,100, 1)
    x_test = x_test.reshape(-1, 100,100, 1)
    x_train.shape, x_test.shape

Reshape our features so we can use them for our next step.

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Scaling features menjadi 0 hingga 1
    x_train = x_train / 255.
    x_test = x_test / 255.
    
    # Menggunakan bantuan OneHotEncoder pada target
    from sklearn.preprocessing import OneHotEncoder
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_train = OneHotEncoder(sparse=False).fit_transform(y_train)
    y_test = OneHotEncoder(sparse=False).fit_transform(y_test)

Once more, change the data type for our features into float32. Do scaling into our features and use OneHotEncoder to our target. OneHotEncoder will change the target into unique category such as [1, 0] for cat and [0, 1] for dog.

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
    
    # Membuat Model CNN dengan 3 layer Convolusi + Relu + Maxpooling
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape =x_train.shape[1:], padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2), strides=2, padding="same"))
    
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2), strides=2, padding="same"))
    
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization()) # Batch Normalization
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2), strides=2, padding="same"))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    model.summary()
    
    # loss dengan categorial crossentropy dan optimizer dengan adam
    model.compile(loss="categorical_crossentropy",
                 optimizer="adam",
                 metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.125)

Import Keras from Tensorflow and build our model for this classification. I was using 3 layers of Convolution 2D with 3×3 kernel size. 16 Channels for first layer, 32 channels for second layer and 64 channels for third layer. I was using Relu as the activation layer and maxpooling layer right after each of convolution layer. As an addition I added Batch Normalization layer right after the third layer of convolution.

Then flatten the result, do dropout to increase our accuracy and do fully connected layer twice. I used Relu for the first activation layer and Sigmoid for the last one. Fit the model with adam optimizer and categorial crossentropy as the loss calculation and store them into a variable named history.

Here is the code where we predict our testing dataset and also the result after we test the model using testing dataset in confusion matrix

    # Print hasil prediksi dengan data test
    import numpy as np
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(x_test)
    y_pred = np.asarray(tf.argmax(y_pred, axis = 1))
    y_test = np.asarray(tf.argmax(y_test, axis = 1))
    y_pred, y_test
    
    # Print Confusion Matrix
    print(confusion_matrix(y_test, y_pred))

![image](https://github.com/radiadus/Cat-and-Dog-Image-Classification/assets/55176713/08613014-50bb-45af-80f9-61c88868c1cf)

_(Confusion Matrix)_

    # Print Skor Akurasi Testing
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))

![image](https://github.com/radiadus/Cat-and-Dog-Image-Classification/assets/55176713/1f108430-f04d-4942-afb2-083e46cabea1)

_(Accuracy Score)_

Last but not least, lets plot the accuracy, validation accuracy, loss, and validation loss for our model.

    # Plotting Akurasi testing dan Akurasi validasi 
    plt.plot(history.history['accuracy'], color="red")
    plt.plot(history.history['val_accuracy'], color='green')
    plt.axis([0,20,0,1])
    plt.show

![image-18](https://github.com/radiadus/Cat-and-Dog-Image-Classification/assets/55176713/9dab3ce1-c78a-49d5-9525-3c1bc3a7fb70)

_(Accuracy Plot)_

    # Plotting loss testing dan loss validasi
    plt.plot(history.history['loss'], color="blue")
    plt.plot(history.history['val_loss'], color='gray')
    plt.show

![image-19](https://github.com/radiadus/Cat-and-Dog-Image-Classification/assets/55176713/a25aaf63-eb88-4673-9ae3-8250637d5087)

_(Loss Plot)_

As the result, I got 95% of accuracy for the training dataset and around 83% for the validation and testing dataset. From confusion matrix we can see that the model was correctly predict 4154 images from total 4990 images. As for the conclusion, 3 layers of convolution layer are very impressive and already enough to get high accuracy prediction. The accuracy may increase if we add more convolution layer but it will cost much more time.
