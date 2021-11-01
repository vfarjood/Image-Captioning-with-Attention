import os
import spacy
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


class TextManager:
    """Class that will build vocabulary and tokenization utilities to be used by natural network model"""

    def __init__(self, freq_threshold=None):
        """Initialization:

        Args:
            freq_threshold: specify the threshold for frequency of a word to be added in vocabulary 
        """

        # create the vocabulary with some pre-reserved tokens: 
        self.vocabulary = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        #PAD: padding , SOS: start of sentence , EOS: end of sentence , UNK: unknown word

        self.spacy_en = spacy.load("en") # for a better text tokenization purpose
        self.freq_threshold = freq_threshold# threshold of word frequency


    def __len__(self):
        """The total number of words in the vocabulary."""

        return len(self.vocabulary)


    def get_key(self, val):
        """Method to be used to get the key of a value in vocabulary
        Args:
            val: the index of the corresponding word in vocabulary

        Return: 
            key: the word corresponding to the given index of the vocabulary
        """

        for key, value in self.vocabulary.items():
            if val == value:
                return key

    def tokenize(self, text):
        """Method for text tokenization (using spacy)"""

        return [token.text.lower() for token in self.spacy_en.tokenizer(text)]

    def build_vocab(self, sentence_list):
        """Method used to build the vocabulary """

        frequencies = Counter() #using 'Counter()' for counting the words
        idx = 4 #index of the first word of the sentence(0,1,2,3 are reserved!)

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                # add the word to the vocab if it reaches minimum frequency threshold
                if frequencies[word] == self.freq_threshold:
                    self.vocabulary[word] = idx
                    idx += 1

    def numericalize(self, text):
        """Method used for converting tokenized text into indices (using spacy)
            Args:
                text: input list of tokenized word

            Return: a list of indices corresponding to each word

        """
        tokenized_text = self.tokenize(text)
        return [self.vocabulary[token] if token in self.vocabulary else self.vocabulary["<UNK>"] for token in tokenized_text]
        
    def save(self, file):
        """Save the vocabulary to file 'vocabulary.dat' """

        with open(file, 'w+', encoding='utf-8') as f:
            for word, idx in self.vocabulary.items():
                f.write("{} {}\n".format(idx, word))

    def load(self, file):
        """Load the vocabulary file 'vocabulary.dat' """

        self.vocabulary = {}

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line_fields = line.split()
                self.vocabulary[line_fields[1]] = int(line_fields[0])

    def load_embedding_file(self, embed_file):
        """ Creates an embedding matrix for the vocabulary.
        Args:
            embed_file: embeddings file with GloVe format
        :return: 
            embeddings tensor in the same order as the words in the vocabulary

        """

        if _data_set.vocab.vocabulary is None:
            raise ValueError("Vocabulary doesn't exists!!!")

        # Find embedding dimension
        with open(embed_file, 'r') as f:
            embed_size = len(f.readline().split()) - 1
    
        words_in_vocab = set(_data_set.vocab.vocabulary.keys()) # create a copy of words from vocabulary into a set
    
        # Create the initialized tensor for embeddings
        embedding_matrix = torch.FloatTensor(len(words_in_vocab), embed_size)

        std = np.sqrt(5.0 / embedding_matrix.size(1))# used uniform distribution specially for pre-reserved tokens
        torch.nn.init.uniform_(embedding_matrix, -std, std)

        for line in open(embed_file, 'r'):
            line_fields = line.split()
    
            embed_word = line_fields[0].lower() #force words to be in lower case

            if embed_word not in words_in_vocab: # Ignoring the word which does not exists in the vocabulary
                continue

            word_id = _data_set.vocab.vocabulary[embed_word]
            embedding_matrix[word_id, :] = torch.tensor([float(i) for i in line_fields[1:]], dtype=torch.float32)
    
        return embedding_matrix, embed_size

class CapDataset(Dataset):
    """ Image Captioning Dataset Class which makes a generic dataset for images and captions"""

    def __init__(self, path, captions_file, preprocess_custom=None, empty_dataset=None, eval=None, test=None):
        """ create a dataset:
        Args:
            path: this is a string to the address of dataset directory
            captions_file: this is a string indication the name of captions file
            preprocess_custom: bool value indicating if we want a custom preprocess(True) or a default preprocess(None)
            empty_dataset: bool value indication if we are going to create an empty dataset or not
            eval: bool value indicating if we want to create a dataset for evaluation
            test: bool value indicating if we want to create a dataset for testing
        Returns:

        """

        #initialize attributes
        self.root_dir = path #path to the root directory of dataset
        self.images = [] #holding the images loading from files
        self.captions = [] #holding the captions loading from files

        self.preprocess = None #type of preprocessing for images
        self.data_mean = torch.zeros(3) #create a zeros tensor for mean
        self.data_std = torch.ones(3) #create a ones tensor for standard deviation
        self.data_mean[:] = torch.tensor([0.485, 0.456, 0.406]) #mean which is used by resnet50
        self.data_std[:] = torch.tensor([0.229, 0.224, 0.225]) #std which is used by resnet50
        
        #check whether the path is correct or not
        if path is None:
            raise ValueError("You must specify the dataset path!")
        if not os.path.exists(path) or os.path.isfile(path):
            raise ValueError("Invalid data path: " + str(path))

        #check whether we want a custom preprocessing or just use a default operation
        if preprocess_custom is not None: #using a custom preprocess
            self.preprocess = T.Compose([
                T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=self.data_mean, std=self.data_std),
                ])
        #using default preprocess
        else: 
            self.preprocess = T.Compose([
                T.Resize(226),
                T.RandomCrop(224),
                T.ToTensor(),
                T.Normalize(mean=self.data_mean, std=self.data_std)
                ])


        #create a dataset for training purpose
        if not eval and not test:
            if not empty_dataset:
                # Get image and caption colum from the dataframe
                self.df = pd.read_csv(captions_file)
                self.files = self.df.values.tolist()
                self.images = (self.df["image"]).values.tolist()
                self.captions = (self.df["caption"]).values.tolist()
                # Initialize vocabulary object and build vocab
                self.vocab = TextManager(args.freq_threshold)
                self.vocab.build_vocab(self.captions)
                self.vocab.save('vocabulary.dat')
                print("Vocabulary size: ", len(self.vocab))
            # if we want to create an empty dataset(for splitting purpose)
            else: 
                self.vocab = _data_set.vocab
                self.files = _data_set.files

        # create a dataset in which you only want to evaluate a pre_trained model
        elif eval: 
            if not os.path.exists('vocabulary.dat'):
                raise ValueError("There is no 'Vocabulary.dat' file")
            self.df = pd.read_csv(captions_file)
            self.files = self.df.values.tolist()
            self.images = (self.df["image"]).values.tolist()
            self.captions = (self.df["caption"]).values.tolist()
            self.vocab = TextManager()
            self.vocab.load('vocabulary.dat')
            print("vocabulary is loaded!!  size=", len(self.vocab))

        # create a dataset in which you only want to test a pre_trained model
        elif test:
            if not os.path.exists('vocabulary.dat'):
                raise ValueError("There is no 'Vocabulary.dat' file")
            folder_contents = os.listdir(self.root_dir)
            self.files = [f for f in folder_contents 
                                if os.path.isfile(os.path.join(self.root_dir, f)) and f.endswith(".jpg")]
            for i in range(0, len(self.files)):
                self.images.append((self.files[i]))
                self.captions.extend(["dummy dummy dummy dummy"])
            self.vocab = TextManager()
            self.vocab.load('vocabulary.dat')
            print("vocabulary is loaded!!  size=", len(self.vocab))


    def __len__(self):
        """Compute the lenght of the data set(each image has 5 captions so each iamge is repeated 5 times)"""
        return len(self.captions)

    def __getitem__(self, idx):
        """Load the next (image,caption) from disk.
        Args:
            index: the index of the element to be loaded.
        Returns:
            The image, caption, and caption lenght.
        """

        caption = self.captions[idx]
        img_name = self.images[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        #apply the transfromation to the image
        img = self.preprocess(img)

        #numericalize the caption text
        caption_vec = []
        #add the SOS(start of a sentence)token at the begining of sentence
        caption_vec += [self.vocab.vocabulary["<SOS>"]] 
        caption_vec += self.vocab.numericalize(caption)
        #add the EOS(end of a sentence)token at the begining of sentence
        caption_vec += [self.vocab.vocabulary["<EOS>"]] 
        caption_len = torch.LongTensor([len(caption_vec)])

        return img, torch.tensor(caption_vec), caption_len

    def randomize_data(self):
        """Randomize the dataset."""

        order = [a for a in range(len(self.files))]
        random.shuffle(order)
        # shuffling the data
        self.files = [self.files[a] for a in order]

    def create_splits(self, proportions: list):
        """ split the dataset into our customize proportion

            Args:
                proportions: a list in which we are going to split the dataset

            returns:
                it returns a list of Dataset Objects(one Dataset per split)

        """
        p = 0.0
        invalid_prop_found = False

        for prop in proportions:
            if prop <= 0.0:
                invalid_prop_found = True
                break

            p +=prop
        if invalid_prop_found or p > 1.0 or len(proportions) == 0: # you are allowed to choose a portion of the data(good!!)
            raise ValueError("Invalid fraction for splitting!!!(It must be possitive and its sum must not be grater than 1)")

        data_size = len(self.files)
        num_splits = len(proportions)
        datasets = []
        for i in range(0, num_splits):
            if i==0:
                datasets.append(CapDataset(self.root_dir, self.files, preprocess_custom=args.preprocess, empty_dataset=True))
            elif i==1:
                datasets.append(CapDataset(self.root_dir, self.files, preprocess_custom=None, empty_dataset=True))
            else:
                datasets.append(CapDataset(self.root_dir, self.files, preprocess_custom=None, empty_dataset=True))

        start = 0
        for i in range(0, num_splits):
            p = proportions[i]
            n = int(p * data_size)
            end = start + n 

            datasets[i].images.extend([self.images[z]for z in range(start, end)])
            datasets[i].captions.extend([self.captions[z]for z in range(start, end)])
            start = end

        print("Dataset size:", data_size)
        print("Selected Dataset size:", len(datasets[0]) + len(datasets[1]) + len(datasets[2]) )
        print("-Training size:", len(datasets[0]))
        print("-Validation size:", len(datasets[1]))
        print("-Test size:", len(datasets[2]))
        print("-------------------------------")

        return datasets

    def CapCollate(self, batch):
        """Collate to apply the padding to the captions with dataloader
        Args:
            batch: compose of images, caption, captions lenght 
        Returns:
            images, padded captions, and captions lenght
        """
        self.pad_idx = self.vocab.vocabulary["<PAD>"]
    
        imgs = [item[0].unsqueeze(0) for item in batch] #get the images of the batch 
        imgs = torch.cat(imgs, dim=0)
    
        targets = [item[1] for item in batch] #get the captions of the batch
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
    
        caption_len = torch.LongTensor([item[2].item() for item in batch]) #create a tensor of captions lenght
    
        return imgs, targets, caption_len

class Encoder_CNN(nn.Module):
    """Class that models the CNN for resnet50 in order to encode the input images"""

    def __init__(self):
        super(Encoder_CNN, self).__init__()

        self.resnet = torch.hub.load('pytorch/vision:v0.7.0', 'resnet50', pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        module = list(self.resnet.children())[:-2] #removing the last: AdaptiveAvgPool2d(), Linear(in=2048, out=1000)
        self.resnet = nn.Sequential(*module)

    def forward(self, images):

        features = self.resnet(images)  #(batch,2048,7,7)  Number Of feature Maps=2048, each feature size=7x7

        features = features.permute(0, 2, 3, 1)  #(batch,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1))  #(batch,49,2048)

        return features

class Attention(nn.Module):
    """Class that models the Attention neural model """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        #Linear transformation from decoder and encoder to attention dimension
        self.LinearEncoder = nn.Linear(encoder_dim, attention_dim)
        self.LinearDecoder = nn.Linear(decoder_dim, attention_dim) 

        #Final projection
        self.relu = nn.ReLU()
        self.LinearAttention = nn.Linear(attention_dim, 1)
        
    def forward(self, features, hidden_state):

        #linear transformation from encoder(CNN) and decoder(LSTM) dimension to attention dimension
        LinearEncoder_outputs = self.LinearEncoder(features)
        LinearDecoder_outputs = self.LinearDecoder(hidden_state) 

        #combine encoder and decoder outputs together (we can also use tanh())
        combined_output = self.relu(LinearEncoder_outputs + LinearDecoder_outputs.unsqueeze(1))

        #compute the outpout of attention network
        attention_outputs = self.LinearAttention(combined_output)

        #compute the alphas
        attention_scores = attention_outputs.squeeze(2) 
        alphas = F.softmax(attention_scores, dim=1) #the sum of all alphas should be equal to one.

        #compoute the context vector
        context_vector = features * alphas.unsqueeze(2)
        context_vector = context_vector.sum(dim=1)

        return alphas, context_vector

class Decoder_LSTM(nn.Module):
    def __init__(self, embed_size, pretrained_embed, vocab_size, attention_dim, encoder_dim, decoder_dim):
        super().__init__()

        # check whether to use pretrained word embedding or not
        if pretrained_embed is None:
            embed_size = embed_size
            self.embedding = nn.Embedding(vocab_size, embed_size)
        else:
            embed_size = pretrained_embed.shape[1]
            self.embedding = nn.Embedding.from_pretrained(pretrained_embed, freeze=True)

        # set the model attributes
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        #create attention object
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        #create initial hidden and cell state
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        #create LSTMCell object for decoder
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)

        #Linear transformation for attention output
        self.decoder_out = nn.Linear(decoder_dim, self.vocab_size)
        self.dropout = nn.Dropout(0.3)

        #initialize model parameters randomely in range(-1, 1)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features, captions, caption_len):
        #sort the captions and images corresponding to their lenght
        caption_lengths, sorted_indices = caption_len.sort(dim=0, descending=True)
        features = features[sorted_indices,:,:] #[32, 49, 2048]
        captions = captions[sorted_indices,:]

        seq_length = caption_lengths - 1 # Exclude the last one (the <end> position)
        batch_size = captions.size(0)
        num_features = features.size(1)

        #from sequences of token IDs to sequences of word embeddings
        embeds = self.embedding(captions)

        # Initialize LSTM hidden and cell states
        h, c = self.init_hidden_state(features)

        #create zeros tensor for outputs and alphas
        outputs = torch.zeros(batch_size, seq_length[0], self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length[0], num_features).to(device)

        # for each time step we will ask attention model to returns a context vector
        # based on decoder's previous hidden state then we will generate new word
        for word in range(0, seq_length[0].item()):

            #get context vector with the encoder features and previous hidden state
            alpha, context = self.attention(features, h) # context=[32, 2048]

            #combine embedding vector of the word with context vector and feed it to lstmcell
            lstm_input = torch.cat([embeds[:, word], context], dim=1) # [32, 2348]
            h, c = self.lstm_cell(lstm_input, (h, c))

            #get the logits of the decoder (also we used dropout for regularization purpose)
            logits = self.decoder_out(self.dropout(h))

            #append all generated words in the outputs tensor
            outputs[:, word] = logits
            alphas[:, word] = alpha

        return outputs, alphas, captions, seq_length

    def CapGenerator(self, features, max_len=20, vocab=None):
        """ Method used to generate a caption for a given image """
 
        alphas = []
        captions = []
        batch_size = features.size(0)

        #generate initial hidden state
        h, c = self.init_hidden_state(features)

        # create the initial sentence with starting token
        word = torch.tensor(vocab.vocabulary['<SOS>']).view(1, -1).to(device)
        embeds = self.embedding(word)
        
        #loop for iterating over the maximum sentence lenght to be generated
        for i in range(max_len):

            #given the image to attention model it returns the context vector
            alpha, context = self.attention(features, h)

            #storing the alphas score for loss function
            alphas.append(alpha.cpu().detach().numpy())

            #generating the next word
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.decoder_out(self.dropout(h))
            output = output.view(batch_size, -1)

            # select the word with highest value
            index_of_predicted_word = output.argmax(dim=1)

            # save the generated word into a list
            captions.append(index_of_predicted_word.item())

            # check to stop generation if it predicted <EOS>
            if index_of_predicted_word.item() == 2: # 2 is the index of <EOS> in the vocabulary
                break

            # send back the generated word as the next caption
            embeds = self.embedding(index_of_predicted_word.unsqueeze(0))

        # covert the index of tokens into words
        return [vocab.get_key(idx) for idx in captions], alphas

    def init_hidden_state(self, encoder_out):
        """ Method used for initial state for the models
        Args:
            encoder_out: this is the output of our encoder which we used it here to make an initial state,
                         it is a tensor of dimension (batch_size, num_pixels, encoder_dim)
        Return:
            h: hidden state with the dimension equal to decoder dimension
            c: cell state with output dimension size of decoder
        """
        #get the mean of encoder output units
        mean_encoder_out = encoder_out.mean(dim=1)

        #get the hidden and cell state by means of a linear transformation from encoder dim to decoder dim
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        return h, c

class Encoder_Decoder(nn.Module):
    """ class that create the main model """
    def __init__(self, embed_size, pretrained_embed, vocab_size, attention_dim, encoder_dim, decoder_dim):
        super().__init__()

        self.encoder = Encoder_CNN()
        self.decoder = Decoder_LSTM(
            embed_size=embed_size,
            pretrained_embed = pretrained_embed,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim, 
            )
        #define the loss function to be used
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracies = None
        self.valid_accuracies = None


    def forward(self, images, captions, caption_len):

        #feed the images to the encoder in order to get the features vector
        features = self.encoder(images)

        #feed images and captions with their lenght to the decoder
        outputs, alphas, captions, seq_length = self.decoder(features, captions, caption_len)

        return outputs, alphas, captions, seq_length


    def Train_model(self, train_set, valid_set, num_epochs, learning_rate, resume=None):
        """main method used to train the network""" 
        
        #set the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #ensuring that the model is on training mode
        model.train()

        #check if we are going to resume the previously training state or not
        if resume:
            print("Resuming last model...")
            start_epoch = model.resume('attention_model_state.pth', optimizer)
        else:
            start_epoch=0

        #initialize some parameterz to be used during training
        vocab_size = len(_data_set.vocab)
        epoch_loss = 100
        epoch_acc = 0
        best_val_acc = -1 # the best accuracy computed on the validation data
        self.train_accuracies = np.zeros(num_epochs)
        self.valid_accuracies = np.zeros(num_epochs)
    
        #loop over the epoches
        for epoch in range(start_epoch, num_epochs):

            train_tot_acc = 0 #total accuracy computed on training set
            train_tot_loss = 0 #total loss computed on training set
            num_batch = 0

            for idx, (image, captions, caption_len) in enumerate(iter(train_set)):
    
                image, captions, caption_len = image.to(device), captions.to(device), caption_len.to(device)
    
                # Zero the gradients.
                optimizer.zero_grad()
    
                # Feed forward the data to the main model
                outputs, alphas, captions, seq_length = model(image, captions, caption_len)
                targets = captions[:, 1:] #skip the start token (<SOS>)
                
                #skip the padded sequences
                outputs = pack_padded_sequence(outputs, seq_length.cpu().numpy(), batch_first=True)
                targets= pack_padded_sequence(targets, seq_length.cpu().numpy(), batch_first=True)

                #compute the accuracy of the model
                acc = model.__performance(outputs, targets)

                #compute the loss
                loss = model.__loss(outputs, targets)

                #try to minimize the difference between 1 and the sum of a pixel's weights across all timesteps
                loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()

                #compute backward.
                loss.backward()
    
                #Update the parameters.
                optimizer.step()
    
                print("-train-minibatch: {} loss: {:.5f}".format(idx+1, loss.item()))
                train_tot_acc += acc
                train_tot_loss += loss
                num_batch += 1

            #try to plot an image with its corresponding generated captions on training set
            model.Test_and_Plot(train_set)

            #compute trainin loss and accuracy (average)
            train_avg_acc = train_tot_acc/num_batch
            train_avg_loss = train_tot_loss/num_batch
            print("Average ---> acc: {:.2f} loss: {:.5f}".format(train_avg_acc, train_avg_loss))
            print("---------------------------------------------")

            #evaluate the current model on validation set 
            valid_avg_acc, valid_avg_loss = model.Evaluate_model(valid_set)
            print("Average ---> acc: {:.2f} loss: {:.5f}".format(valid_avg_acc, valid_avg_loss))
            print("---------------------------------------------")

            #save the current model if it is the best
            if valid_avg_acc > best_val_acc:
                best_val_acc = valid_avg_acc
                best_epoch = epoch+1
                model.save(model, optimizer, epoch)

            self.train_accuracies[epoch] = train_avg_acc
            self.valid_accuracies[epoch] = valid_avg_acc

            print(("Epoch={}/{}:  Tr_loss:{:.5f}  Tr_acc:{:.2f}  Va_acc:{:.2f}" + 
                   (" ---> **BEST**" if best_epoch == epoch + 1 else ""))
                    .format(epoch+1, num_epochs, train_avg_loss, train_avg_acc, valid_avg_acc))
            print("---------------------------------------------")

        #save the final model
        torch.save(model, 'attention_model.pth')
    

    def Evaluate_model(self, valid_set):

        #set the model on evaluating mode
        model.eval()

        #set some initial parameteres
        tot_acc = 0
        tot_loss = 0
        num_batch = 0
    
        for idx, (image, captions, caption_len) in enumerate(iter(valid_set)):
            image, captions, caption_len = image.to(device), captions.to(device), caption_len.to(device)

            #call the main model to generate the captions
            outputs, alphas, captions, seq_length = model(image, captions, caption_len)
            targets = captions[:, 1:] #skip the first token (SOS)

            #skip the padded sequences
            outputs = pack_padded_sequence(outputs, seq_length.cpu().numpy(), batch_first=True)
            targets = pack_padded_sequence(targets, seq_length.cpu().numpy(), batch_first=True)

            #compute the accuracy and loss
            acc = model.__performance(outputs, targets)
            loss = model.__loss(outputs, targets)
            loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()

            #update parameters used during evaluation
            tot_acc += acc
            tot_loss += loss
            num_batch +=1
            print("-valid-minibatch: {} loss: {:.5f}".format(idx+1, loss.item()))

        #generate a caption for an image from validation set to show the accuracy
        model.Test_and_Plot(valid_set)

        #compute the accuracy and loss of validation set (average)
        avg_acc = tot_acc/num_batch
        avg_loss = tot_loss/num_batch
        model.train() #set the model back to training mode
    
        return avg_acc, avg_loss


    def Test_and_Plot(self, test_data, attention=None):
        """Method used to plot the image with its corresponding generated caption """

        model.eval()

        with torch.no_grad():
            dataiter = iter(test_data)
            img, _, _ = next(dataiter)
            features = model.encoder(img[0:1].to(device))
            caps, alphas = model.decoder.CapGenerator(features, vocab=_data_set.vocab)
            caption = ' '.join(caps)
            show_image(img[0], title=caption)

            if attention:
                plot_attention(img[0], caps, alphas)


    def __loss(self, outputs, targets):
        """ function to be used for computing the loss """ 

        loss = self.criterion(outputs.data, targets.data)
        return loss


    def __performance(self, outputs, targets):
        """function to be used for computing the performance of the model """ 

        #returns the index of the word with the highst value
        highest_indices = outputs.data.argmax(dim=1)
        highest_indices = highest_indices.reshape(-1, 1)

        #check if the predicted output is equal to the targets
        word_correct = highest_indices.eq(targets.data.view(-1,1))
        seq_correct = word_correct.float().sum()

        #compute the batch accuracy
        acc = seq_correct.item() * (100.0/targets.data.shape[0])

        return acc
    

    def save(self, model, optimizer, num_epochs):
        """save the model"""
        checkpoint = {
            'num_epochs':num_epochs,
            'optimizer': optimizer.state_dict(),
            'model_state':model.state_dict()
            }
        torch.save(checkpoint,'attention_model_state.pth')
        return
    
    def resume(self, checkpoint, optimizer):
        """resume the model"""
        checkpoint = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint['num_epochs']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return start_epoch


def plot_attention(img, result, attention_plot):
    # recover the original image from transformed image
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7, 7)

        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

def show_image(img, title=None):
    """Imshow for Tensor."""
    # unnormalize
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def save_acc_graph(train_accs, valid_accs):
    """Plot the accuracies of the training and validation data computed during the training stage.

    Args:
        train_accs,valid_accs: the arrays with the training and validation accuracies (same length).
    """

    plt.figure().clear()
    plt.clf()
    plt.close()
    plt.figure()
    plt.plot(train_accs, label='Training Data')
    plt.plot(valid_accs, label='Validation Data')
    plt.ylabel('Accuracy %')
    plt.xlabel('Epochs')
    plt.ylim((0, 100))
    plt.legend(loc='lower right')
    plt.savefig('training_stage.pdf')
    plt.figure().clear()
    plt.close()
    plt.clf()

def parse_command_line_arguments():
    """Parse command line arguments and checking their values"""

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('mode', type=str, choices=['train', 'eval', 'test'],
                        help='train or evaluate or test the model')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from previouse training phase( 1:Yes or 0:No) default: 0')
    parser.add_argument('data_location', type=str,
                        help='define training_set or test_set directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate (Adam) (default: 3e-4)')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of working units used to load the data (default: 0)')
    parser.add_argument('--freq_threshold', type=int, default=1,
                        help='threshold for word frequencies (default: 1)')
    parser.add_argument('--randomize', type=str, default=None,
                        help='shuffling the data set before splitting (1:Yes or 0:No) default: 1 ')
    parser.add_argument('--preprocess', type=str, default=None,
                        help='choose a customize preprocess {default or custom} default: default ')
    parser.add_argument('--splits', type=str, default='0.04-0.008-0.008',
                        help='fraction of data to be used in train set and val set (default: 0.7-0.3)')
    parser.add_argument('--glove_embeddings', type=str, default=None,
                        help='pre-trained embeddings file will be loaded (default: None)')  
    parser.add_argument('--embed_size', type=int, default=128,
                        help='word embedding size (default: 128)')
    parser.add_argument('--attention_dim', type=int, default=256,
                        help='input dimension of attention model (default: 256)')
    parser.add_argument('--encoder_dim', type=int, default=2048,
                        help='input dimension of encoder model (default: 2048)')
    parser.add_argument('--decoder_dim', type=int, default=512,
                        help='input dimension of decoder model (default: 512)')
    parser.add_argument('--device', default='gpu', type=str,
                        help='device to be used for computations {cpu, gpu} default: gpu')

    parsed_arguments = parser.parse_args()

    #converting split fraction string to a list of floating point values ('0.7-0.15-0.15' => [0.7, 0.15, 0.15])
    splits_string = str(parsed_arguments.splits)
    fractions_string = splits_string.split('-')
    if len(fractions_string) != 3:
        raise ValueError("Invalid split fractions were provided. Required format (example): 0.7-0.15-0.15")
    else:
        splits = []
        frac_sum = 0.
        for fraction in fractions_string:
            try:
                splits.append(float(fraction))
                frac_sum += splits[-1]
            except ValueError:
                raise ValueError("Invalid split fractions were provided. Required format (example): 0.7-0.15-0.15")
        if frac_sum > 1.0 or frac_sum < 0.0:
            raise ValueError("Invalid split fractions were provided. They must sum to 1.")

    # updating the 'splits' argument
    parsed_arguments.splits = splits

    return parsed_arguments


""" STARTING FROM HERE """

if __name__ == "__main__":
    args = parse_command_line_arguments()

    print("\n-------------------------------")
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))


    # *** TRAINING *** 
    # if you have choosed 'train' then it will start to train the model from here.

    print("-------------------------------")
    if args.mode == 'train':

        if args.device == 'gpu':
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f'There are {torch.cuda.device_count()} GPU(s) available.')
                print('Using device:', torch.cuda.get_device_name(0))
            else:
                print('GPU is not availabe!! Using device: CPU')
                device = torch.device("cpu")
        else:
            print('Using device: CPU')
            device = torch.device("cpu")
        
        #create the dataset object
        _data_set = CapDataset(path=args.data_location + "/Images",
                                  captions_file=args.data_location + "/captions.txt",
                                  preprocess_custom=args.preprocess)

        #check if you need to randomize the data or not
        if args.randomize:
            _data_set.randomize_data()

        #split the data into three parts:
        [_train_set, _val_set, _test_set] = _data_set.create_splits(args.splits)

        #check whether using pretrained embeddings or not
        if args.glove_embeddings is not None:
            _pretrained_embed, embed_size = _data_set.vocab.load_embedding_file(args.glove_embeddings)
            print("Loading embeddings: DONE")
            print("Embeddding_size= ", embed_size)
            print("-------------------------------")
        else:
            _pretrained_embed = None

        #create dataloader for training set
        _train_set = DataLoader(dataset=_train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                                        collate_fn=_data_set.CapCollate)
        #create dataloader for validation set
        _val_set   = DataLoader(dataset=_val_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                                        collate_fn=_data_set.CapCollate)
        #create dataloader for test set
        _test_set  = DataLoader(dataset=_test_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                                        collate_fn=_data_set.CapCollate)

        #create the model
        model = Encoder_Decoder( embed_size=args.embed_size,
                                pretrained_embed = _pretrained_embed,
                                vocab_size=len(_data_set.vocab),
                                attention_dim=args.attention_dim,
                                encoder_dim=args.encoder_dim,
                                decoder_dim=args.decoder_dim,
                               ).to(device)

        #starting to train the model
        print("-------------------------------")
        print("\nTraining The Model...")
        model.Train_model(_train_set, _val_set, args.epochs, args.learning_rate, args.resume)
        save_acc_graph(model.train_accuracies, model.valid_accuracies)

        #starting to evaluate the model (on trainin-set, validation-set, test-set)
        print("-------------------------------")
        print("\nEvaluating The Model...")

        print("\n-On training_set...")
        train_acc, train_loss = model.Evaluate_model(_train_set)
        print("Average On Training ---> acc:{:.2f}  loss:{:.5f}".format(train_acc, train_loss))

        print("\n-On validation_set...")
        val_acc, val_loss = model.Evaluate_model(_val_set)
        print("Average On Validation ---> acc:{:.2f}  loss:{:.5f}".format(val_acc, val_loss))

        print("\n-On test_set...")
        test_acc, test_loss = model.Evaluate_model(_test_set)
        print("Average On Test ---> acc:{:.2f}  loss:{:.5f}".format(test_acc, test_loss))


    # *** EVALUATE *** 
    # if you have choosed 'test' then it will evaluate the model starting from here:

    elif args.mode == 'eval':
        print("\nEvaluating The Model...")

        if args.device == 'gpu':
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f'There are {torch.cuda.device_count()} GPU(s) available.')
                print('Using device::', torch.cuda.get_device_name(0))
            else:
                print('GPU is not availabe!! Using device: CPU')
                device = torch.device("cpu")
        else:
            print('Using device: CPU')
            device = torch.device("cpu")

        #create the dataset object
        _data_set = CapDataset(path=args.data_location + "/Images",
                                  captions_file=args.data_location + "/captions.txt",
                                   empty_dataset=True, eval=True)

        #create dataloader for the data set to be validate the model
        _val_set  = DataLoader(dataset=_data_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                                        collate_fn=_data_set.CapCollate)

        # check whether using pretrained embeddings or not
        if args.glove_embeddings is not None:
            _pretrained_embed, embed_size = _data_set.vocab.load_embedding_file(args.glove_embeddings)
            print("Loading embeddings: DONE")
            print("Embeddding_size= ", embed_size)
            print("-------------------------------")
        else:
            _pretrained_embed = None

        #create the initial model
        model = Encoder_Decoder( embed_size=args.embed_size,
                                pretrained_embed = _pretrained_embed,
                                vocab_size=len(_data_set.vocab),
                                attention_dim=args.attention_dim,
                                encoder_dim=args.encoder_dim,
                                decoder_dim=args.decoder_dim
                               ).to(device)

        #check the path to load the model in which you want to evaluate it
        if not os.path.exists('attention_model.pth'):
            raise ValueError("There is no 'attention_model.pth' file!!!")
        else:
            print("-------------------------------")
            print("loading the model...")

        #load the model
        model = torch.load('attention_model.pth', map_location=device)

        #start to evaluate the model with the corresponding dataset
        val_acc, val_loss = model.Evaluate_model(_val_set)
        print("Average On Validation ---> acc:{:.2f}  loss:{:.5f}".format(val_acc, val_loss))


    # *** TEST *** 
    # If you have choosed 'test' then it will only test the model starting from here.

    elif args.mode == 'test':
        print("\nTesting The Model...")

        if args.device == 'gpu':
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f'There are {torch.cuda.device_count()} GPU(s) available.')
                print('Using device:', torch.cuda.get_device_name(0))
            else:
                print('GPU is not availabe!! Using device: CPU')
                device = torch.device("cpu")
        else:
            print('Using device: CPU')
            device = torch.device("cpu")

        #create the dataset object 
        _data_set = CapDataset(path=args.data_location + "/Images",
                                  captions_file=args.data_location + "/captions.txt",
                                   empty_dataset=True, test=True)

        #create dataloader for the test set
        _test_dataset  = DataLoader(dataset=_data_set, batch_size=1, num_workers=1, shuffle=False,
                                            collate_fn=_data_set.CapCollate)

        # check whether using pretrained embeddings or not
        if args.glove_embeddings is not None:
            _pretrained_embed, embed_size = _data_set.vocab.load_embedding_file(args.glove_embeddings)
            print("Loading embeddings: DONE")
            print("Embeddding_size= ", embed_size)
            print("-------------------------------")
        else:
            _pretrained_embed = None

        #create the model object
        model = Encoder_Decoder( embed_size=args.embed_size,
                                pretrained_embed = _pretrained_embed,
                                vocab_size=len(_data_set.vocab),
                                attention_dim=args.attention_dim,
                                encoder_dim=args.encoder_dim,
                                decoder_dim=args.decoder_dim
                               ).to(device)
        #check whether the model exists in the path or not
        if not os.path.exists('attention_model.pth'):
            raise ValueError("There is no 'attention_model.pth' file!!!")
        print("-------------------------------")
        model = torch.load('attention_model.pth', map_location=device)

        #set the model to be in evaluation mode
        model.eval()

        #iterate over the test images that you provided
        dataiter = iter(_test_dataset)
        for i in range(0, len(_data_set)):
            model.Test_and_Plot(dataiter, attention=True)


    else:
        raise ValueError("You must specify the operation you need!!! ('train', 'eval', 'test'")

