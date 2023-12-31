**Settings things up:** we can define this in 3 parts

1. **Word embedding:** Each word should be turn into list of numbers so that computer can understand.
2. **LSTM Layer:** It is like main part or brain of this model, It will guess the next word.
3. **Connection:** we need to connect and convert what LSTM guess to an actual human readable guess or result
"""

import torch
import torch.nn as nn
import torch.optim as optim

#Lets defing the language model class
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # this is layer for word emmbedding
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True) #LSTM Layer
        self.fc = nn.Linear(hidden_dim, vocab_size) #connection

    #Forward propagation function
    def forward(self, x, hidden):
        embeds = self.embedding(x) # this line will convert word indices to embeddings
        lstm_out, hidden = self.lstm(embeds, hidden) #LSTM output and also the hidden state
        output = self.fc(lstm_out) # output
        return output, hidden

"""**Preparing and Preprocessing the data**"""

#Sample Text (You can use any dataset)

text = 'Hi, How are you?'
words = text.split()
vocab = set(words)
vocab_size = len(vocab)
words_to_index = {word: index for index, word in enumerate(vocab)} #Mapping words to indices
index_to_word = {index: word for word, index in words_to_index.items()} #Mapping indices to words

"""**Data Preparation**

1. we should break the sample text to mini-text of length of 3 or whatever your seq length like " Hi, how are" or " how are you?" etc.
2. For this mini-text, we should also keep track of the words that comes right after it.
"""

seq_length = 3 # you can adjust the sequence according to your datasets
sequences = []
targets = []

#lets create loop to generate sequences and its target
for i in range(len(words) - seq_length):
    sequence = words[i:i + seq_length]
    target = words[i + seq_length]
    sequences.append([words_to_index[word] for word in sequence])
    targets.append(words_to_index[target])

# Lets create Hyperparameters
embed_dim = 10
hidden_dim = 20
n_layers = 1
learning_rate = 0.01
epochs = 200 # higher is better

# Initialize model, loss and optimizer
model = LanguageModel(vocab_size, embed_dim, hidden_dim, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""**Training Time:**

1. We show these mini-text to the model, and it tries to guess the next word.
2. We can tell the model if it's wrong and make it better(means what is "**loss**" and "**optimize**" it )
3. It will repeat and get better every time(as **epochs** are numbers of times it will repeat and get training)
"""

# Training the model

for epoch in range(epochs):
    hidden = None # this is initial hidden state

    # Loop through each seq
    for sequence, target in zip(sequences, targets):
      input_data = torch.tensor([sequence])
      target_data = torch.tensor([target])

      #Model forward pass
      output, hidden = model(input_data, hidden)

      #detach hidden state to prevent backpropagation
      hidden = (hidden[0].detach(), hidden[1].detach())

      #Calculate the loss
      loss = criterion(output.view(-1, vocab_size), target_data.repeat(seq_length))

      #Backpropagation and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Display loss every 20 epochs
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

"""**Testing:**

After training, we can test it, if it predicts the next word correctly.
"""

with torch.no_grad(): #No gradient tracking during prediction
    test_seq = torch.tensor([sequences[0]]) # take the first seq for the data


    output, _= model(test_seq, None)
    output = output[0][-1] #last prediction

    predicted_index = torch.argmax(output).item()

    # check the predicted word:
    if predicted_index in index_to_word:
        predicted_word = index_to_word[predicted_index]
        print(f" The seq {' '.join([index_to_word[idx] for idx in sequences[0]])}, next predicted word is: {predicted_word}")
    else:
      print(f" Predicted index {predicted_index} is not in data")
