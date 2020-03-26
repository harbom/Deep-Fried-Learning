import pandas as pd

# prepare training data array consisting of meme id's, binary identification of top caption v bottom caption, and the characters in each caption
raw_data = pd.read_csv("Meme_training_data.csv")
#confirm that the csv was read correctly
#print(raw_data.head())

training_data = []

#populate the training_data array
def process_string(string,memeid,topOrBottom):
    
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
test_file = open("check_training_data_arr.txt","w")
for i in training_data:
    print(i,"\n",file = test_file)