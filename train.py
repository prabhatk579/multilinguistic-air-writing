import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import sys

Training = sys.argv[1] # Change this to False if you want to use trained model after initial training
dataset = sys.argv[2] # Set the dataset name

# Read the data...
data = pd.read_csv('data/'+dataset+'.csv').astype('float32')
# Split data the X - Our data , and y - the prdict label
X = data.drop('0',axis = 1)
y = data['0']
# Reshaping the data in csv file so that it can be displayed as an image...

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28))


print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)
print(train_y.shape)
print(test_y.shape)
# Dictionary for getting characters from index values...

if dataset == 'A_Z_Handwritten_Data':         
    model_name = 'eng_alphabets'
    word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
    res = 28
    classes = 26
elif dataset == 'roman_digits':            
    model_name = 'roman_digits'
    word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
    res = 28
    classes = 10
elif dataset == 'hindi_alphabets':            
    model_name = 'hindi_alphabets'
    word_dict = {0: 'CHECK', 1: 'ka', 2: 'kha', 3: 'ga', 4: 'gha', 5: 'kna', 6: 'cha',
            7: 'chha', 8: 'ja', 9: 'jha', 10: 'yna', 11: 'taa', 12: 'thaa', 13: 'daa', 
            14: 'dhaa', 15: 'adna', 16: 'ta', 17: 'tha', 18: 'da', 19: 'dha', 20: 'na', 
            21: 'pa', 22: 'pha', 23: 'ba', 24: 'bha', 25: 'ma', 26: 'yaw', 27: 'ra', 
            28: 'la', 29: 'waw', 30: 'sha', 31: 'sha',32: 'sa', 33: 'ha',
            34: 'kshya', 35: 'tra', 36: 'gya', 37: 'CHECK'}
    # word_dict = {0: 'ka', 1: 'kha', 2: 'ga', 3: 'gha', 4: 'kna', 5: 'cha',
    #         6: 'chha', 7: 'ja', 8: 'jha', 9: 'yna', 10: 'taa', 11: 'thaa', 12: 'daa', 
    #         13: 'dhaa', 14: 'adna', 15: 'ta', 16: 'tha', 17: 'da', 18: 'dha', 19: 'na', 
    #         20: 'pa', 21: 'pha', 22: 'ba', 23: 'bha', 24: 'ma', 25: 'yaw', 26: 'ra', 
    #         27: 'la', 28: 'waw', 39: 'sha', 30: 'sha',31: 'sa', 32: 'ha',
    #         33: 'kshya', 34: 'tra',35: 'sra', 36: 'gya'}
    res = 32
    classes = 38
elif dataset == 'devnagri_digits':             
    model_name = 'devnagri_digits'
    word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
    res = 32
    classes = 10

# Plotting the number of alphabets in the dataset...
train_yint = np.int0(y)
count = np.zeros(classes, dtype='int')
for i in train_yint:
    count[i] +=1

alphabets = []
for i in word_dict.values():
    alphabets.append(i)

fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.barh(alphabets, count)

plt.xlabel("Number of elements ")
plt.ylabel("Alphabets")
plt.grid()
plt.show()
# Shuffling the data ...
shuff = shuffle(train_x[:100])

fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()

for i in range(9):
    axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
plt.show()

# Reshaping the training & test dataset so that it can be put in the model...
train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
print("New shape of train data: ", train_X.shape)

test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
print("New shape of test data: ", test_X.shape)

# Converting the labels to categorical values...
train_yOHE = to_categorical(train_y, num_classes = classes, dtype='int')
print("New shape of train labels: ", train_yOHE.shape)

test_yOHE = to_categorical(test_y, num_classes = classes, dtype='int')
print("New shape of test labels: ", test_yOHE.shape)

# CNN model...
if Training == "True":
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(res,res,1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())

    model.add(Dense(64,activation ="relu"))
    model.add(Dense(128,activation ="relu"))

    model.add(Dense(26,activation ="softmax"))



    # model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


    history = model.fit(train_X, train_yOHE, epochs= 4, callbacks=[reduce_lr, early_stop],  validation_data = (train_X, train_yOHE))


    model.summary()
    model.save('models/model'+model_name+'.h5')

    # Displaying the accuracies & losses for train & validation set...

    print("The validation accuracy is :", history.history['val_accuracy'])
    print("The training accuracy is :", history.history['accuracy'])
    print("The validation loss is :", history.history['val_loss'])
    print("The training loss is :", history.history['loss'])
    
else:
    model = tf.keras.models.load_model('models/model_'+model_name+'.h5')
    model.summary()