import pandas as pd
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
from keras.layers.normalization import BatchNormalization
import numpy as np

# prepare training data array consisting of meme id's, binary identification of top caption v bottom caption, and the characters in each caption
raw_data = pd.read_csv("Meme_training_data.csv")
#confirm that the csv was read correctly
#print(raw_data.head())

training_data = []
character_to_int_mapping = []

#populate the training_data array
def process_string(string,memeid,topOrBottom):
    #append numbers of the memeid to the character_int mapping array
    if memeid not in character_to_int_mapping:
        character_to_int_mapping.append(str(memeid))
    memeid = str(memeid).zfill(4) #pad the memeid with zeroes before hand to make all id's the same length
    
    #if the dataframe held a NaN float value there (intentionally), it was a blank caption
    if (isinstance(string,float)):
        #then, there is only 1 element, with the characters_seen being blank and the next_character being |
        new_entry_first_string = ("%s %d %s" %(memeid,topOrBottom," "))
        next_character = "|"

        training_data.append([new_entry_first_string,next_character])

    else: #go ahead as planned

        characters_seen = ""
        for j in range(len(string)+1):
            #i is the memeid, topOrBottom says 0 if its a top caption, 1 if its a bottom caption
            new_entry_first_string = ("%s %d %s" %(memeid,topOrBottom,characters_seen))
            
            #if the new character is a space, or if the string ended, add |
            next_character = ""
            if j == len(string) or string[j]==" ":
                next_character = "|"
            else:
                next_character = string[j]

            #append to the character to int mapping array
            if not(next_character in character_to_int_mapping):
                character_to_int_mapping.append(next_character)
            #append both strings to a new array, and add that array to training_data
            training_data.append([new_entry_first_string,next_character])

            characters_seen += next_character

#for each image
for i in range(len(raw_data)):
    top_caption = raw_data["Top Caption"][i]
    bottom_caption = raw_data["Bottom Caption"][i]

    #for each character in the either caption, the format is the following:
    # ["{memeID}   {binaryClassifierForToporBottom}  {characters including up to and including that character}","{the next character}"]
    # notice that the entry has 2 strings

    #passing in the string, i==memeid, and the binary classifier for top or bottom string, for both the top and the bottom captions
    process_string(top_caption,i,0)
    process_string(bottom_caption,i,1)


#check if populating array was ok
def check_training_data_arr():
    test_file = open("check_training_data_arr.txt","w")
    for i in training_data:
        print(i,"\n",file = test_file)

#check_training_data_arr()



# tensorize the data, reshape to fit into the CNN
main_texts = [i[0] for i in training_data]
labels = [i[1] for i in training_data]

character_to_int_mapping.append(" ") #spaces weren't ever accounted for when populating training_data array
#print(character_to_int_mapping)

#reshape the texts and lables array to be indices of the character_to_int_mapping array

#transform the main texts
transformed_main_texts = []
for row in main_texts:
    transformed_row = []
    for char in row:
        transformed_row.append(character_to_int_mapping.index(char))
    transformed_main_texts.append(transformed_row)

#transform the labels
transformed_labels = []
for entry in labels: #entry will just be a 1 character string, its the 'next character'
    transformed_labels.append(character_to_int_mapping.index(entry))

#testing
#print(transformed_main_texts[0:70])
#print(transformed_labels)

#pad the main texts with 0's so they're the same length using keras.tf pad_sequences
padding_length = 128
main_data = pad_sequences(transformed_main_texts,maxlen=padding_length)

#print(main_data)[0:70]

#randomize training data (stats :pog:)
indices = np.arange(main_data.shape[0]) #gets array of indices from len(main_data)
np.random.shuffle(indices) #shuffles indices
main_data = main_data[indices] #assigns a new order for the array
for i in range(len(indices)):
    transformed_labels[i] = transformed_labels[indices[i]]

#ratios help the model cross reference it's predictions to memes its not referencing in the training set
ratio = .2 if main_data.shape[0] < 1000000 else .02
num_val_samples = int(ratio * main_data.shape[0])

#split the data based on the ratios
x_train = main_data[:-num_val_samples] #the first to the last {num_val_samples}'th samples
y_train = transformed_labels[:-num_val_samples] 
x_test = main_data[-num_val_samples:] #the {num_val_samples}'th sample to the end
y_test = transformed_labels[-num_val_samples:]

#testing
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

#end preprocessing
#-------------------------------
#model creation

#CNN model
EMBEDDING_DIM = 16

CNNmodel = Sequential()
CNNmodel.add(Embedding(len(character_to_int_mapping) + 1, EMBEDDING_DIM, input_length=padding_length)) #Embedding layer to convert the 128 char long array to 128x16 matrix: helps convert to integers instead of floats
CNNmodel.add(Conv1D(1024, 5, activation='relu', padding='same'))                                       #Conv1D layers with 1024 output filters, kernel size of 5 (length of convolution filter): looks at constructing words from chars
CNNmodel.add(BatchNormalization())                                                                     #BatchNormalization so that the next layers are normalized based on the mean/variance for current layer, helps improving training speed
CNNmodel.add(MaxPooling1D(2))                                                                          #Pooling reduces spatial size of features (by 2 in this case) to help computation in the network
CNNmodel.add(Dropout(0.25))                                                                            #Drops out a random 25% of data to avoid overfit
CNNmodel.add(Conv1D(1024, 5, activation='relu', padding='same'))                                       #relu activation is basically max(0,input), is good for a function that accepts the weighted values and outputs a direct value to an output node
CNNmodel.add(BatchNormalization())                                                                     #same padding ensures that the dimensions of the output layer match the dimensions of the input layer
CNNmodel.add(MaxPooling1D(2))
CNNmodel.add(Dropout(0.25))
CNNmodel.add(Conv1D(1024, 5, activation='relu', padding='same'))
CNNmodel.add(BatchNormalization())
CNNmodel.add(MaxPooling1D(2))
CNNmodel.add(Dropout(0.25))
CNNmodel.add(Conv1D(1024, 5, activation='relu', padding='same'))
CNNmodel.add(BatchNormalization())
CNNmodel.add(MaxPooling1D(2))
CNNmodel.add(Dropout(0.25))
CNNmodel.add(Conv1D(1024, 5, activation='relu', padding='same'))
CNNmodel.add(BatchNormalization())
CNNmodel.add(GlobalMaxPooling1D())                                                                     #GlobalMaxPooling1D also shrinks layer, but automatically chooses how to do it (instead of the param of 2 in previous calls)
CNNmodel.add(Dropout(0.25))
CNNmodel.add(Dense(1024, activation='relu'))
CNNmodel.add(BatchNormalization())
CNNmodel.add(Dropout(0.25))
CNNmodel.add(Dense(len(transformed_labels), activation='softmax'))                                           #Dense houses the activation function, specifies the softmax activation function (nonlinear)

CNNmodel.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])         #compile the CNNmodel: rmsprop optimizer is good optimizer, the loss is specific to our example when we optimize for choosing the best category among 2 labels (top/bottom captions). sparse because the labels are ints between 0 and 70 (good for memory)
#CNNmodel.summary()


#should come back to add a GAN model for cross comparison after we get the CNN model up and running
#end model-creation(?)
#----------------------------------------------------------

#Training time
model_path = "models/conv_model"
num_epochs = 48
batch_size = 256

checkpointer = ModelCheckpoint(filepath=model_path + '/model.h5',verbose=1,save_best_only=True)

#print(len(x_train[0]),len(y_train),len(x_test),len(y_test))
history = CNNmodel.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=num_epochs,batch_size=batch_size,callbacks=[checkpointer])