import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
'''
# Arguments allow you to chnge things like our hyperperameters from the console.
parser = argparse.ArgumentParser(description='This is a demostration program')

# Here you add an argument to the parser specifying the expected type, a help message, etc...
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

args = parser.parse_args()

# Now you can use the argument value in the program...
print(f'batch_size: {args.batch_size}')
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#hyperparameters. These can be changed to reduce the loss.
#the batch size and block size can be adjusted based on available dedicated gpu video card memory
#batch_size = args.batch_size #how many of the batches we want to happen at the same time
batch_size = 32
block_size = 128 #Token sequence length
#common learning rates to use are 1e-3, 1e-4, 3e-3, 3e-4
learning_rate = 3e-4 #learning rate can be experimented on to evaluate which value produces the best prformance and qaulity over time.
max_iterations = 1 #iterations we want to train the model for
eval_iterations = 100 #Used to report loss
n_embd = 384 #the total dimensions we want to capture from all heads concatonated together.
n_layer = 1 #number of decoder blocks you want to have
n_head = 1 #number of heads we have running in parallel
dropout = 0.2

#Look into auto-tuning which can find the optimal parameters for your module through tril and error.


chars = ""
with open('data/vocab.txt', 'r', encoding='utf-8') as file: #opens the input text data file
    text = file.read()
    chars = sorted(list(set(text)))
    
vocab_size = len(chars)


# Tokeniser code. encodes the string values within the data to integer tokens.

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join(int_to_string[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)


#This Cell is responsible for creating data batches for feeding into the model

'''
n = int(0.8*len(data)) #the 0.8 represents 80% of the data
training_data = data[:n]
validation_data = data[n:]
#The chunk of code above is for producing the training set and validation set split
'''

#memory map for using small snippets of text from the single file of any size. This is so all the text doesnt get loaded into memory at once (memory mapping).


# This cell containes the whole architecture of the GPT model (think back to the diagram).

#building Scaled Dot Procuct attention 
class Head(nn.Module):
    """singlar head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #registers the no look ahead masking to the models state so it starts in a masked state
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #input of size (Batch, Time-step, Channels)
        #outpu of size (Batch, Time-step, Head Size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs) shape
        q = self.query(x) # (B,T,hs) shape
        #compute the attention scores for the characters
        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 #Dot products the keys and queries. The transposing allows for the grid multiplication. Flips second last dimensin (-2) with last dimension (-1). The scaling part makes sure no dot product is too overpowering allowing us to hear all parts of the input equally
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #masking to make sure the model cant look ahead at future parameters and cheat. Each time step exposes the next value in the tensor.
        weights = F.softmax(weights, dim=-1) #apply softmax function to get (B, T, T). The softmax makes the model more confident in it's predictions
        weights = self.dropout(weights)
        #performs weighted aggregation of the value
        v = self.value(x) # (B,T,hs)
        out = weights @ v #multiply value weight to softmax value. (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

#building MultiHeadAttention connection within decoder
class MultiHeadAttention(nn.Module):
    """multible heads of self attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #self.heads is a module list. He havehaeds in parallel for each head
        self.proj = nn.Linear(head_size * num_heads, n_embd) #this line project the headds_size miltiplied by the number of heads to an n_embd. Adds more learnable parameters to help model learn more about the given text
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatonating each head together along the last dimension (channel dimension) (B,T,C) -> (B,T,[h1,h1,h1,h1,h2,h2,h2,h2,h3,h3...])
        out = self.dropout(self.proj(out))
        return out

#building feedforward connection within decoder block
class FeedForward(nn.Module):
    """sample linear layer followed by a non-linear layer (in this case ReLU activation) and then another linear layer"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), #makes percentage of neurons drop out and become zero
        )
    def forward(self, x):
        return self.net(x)

#building the decoder blocks
class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        #svsvsvsvsvsv
        super().__init__()
        head_size = n_embd // n_head #n_embed is the number of features we have and n_head is the number of heads we have
        self.sa = MultiHeadAttention(n_head, head_size) #sa is short for self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    #post-norm or pre-norm architecture. We use post-norm here because it converges better for the data and model parameters being used.
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

#Building language model
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #creating an embedding table. This is basically a look-up table. This is a grid with tokens where the probability for a predicted token can be seen.
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # n_embd will be used in positional and token embedding to store tokens in very last vectors.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #no of decoder blocks we run sequentially. The asterix repeats the block part for the number of n_layers
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size) # compressing transformation for making it softmax workable.
        
        #this next chunk of code initialises the weights around a standard deviation of 0.02 so that the initial weights are somewhat consostant
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    #the forward pass function descripes how inputs into the network will be passed trough the layers.
    #.view turns a matrix unpacked into x, y, z, etc... coordinates back into a tensor.
    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(index)
        position_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_embed + position_embed # (B,T,C) shape
        x = self.blocks(x) # feeding in x parameters in (B,T,C) shape
        x = self.ln_f(x) # once the parameters have gone through all layers, they are fed into the final layer norma in (B,T,C) shape
        logits = self.lm_head(x) # use this line to get the parametes into probabilitis which van be fed into the softmax function in (B,T,vocabulary_size) shape
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape #B is for batch, T is for time, Channels is vocabulary size
            logits = logits.view(B*T, C) #because we're paying attention to the vocabulary, or the Channles we can blent the batch and time. As long a logits and targets have the same batch and time then this will be fine.
            #B and T are multiplied because Pytorch expects an input shape of B by C by etc... (e.g B, C, T) but the shape is B by T by C (B, T, C), so the B and T are combined into one.
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #cross_entropy is a loss function

        return logits, loss #logits are basically a bunch of floating point numbers which are normalised. They show the contribution of a single token to the whole embedding, basically a probability distribution for what you want to predict

    def generate(self, index, max_new_tokens): #max_new_tokens indicates the max length of the generated text/tokens
        # index is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop index to the last block_size tokens (block size specified at beginning)
            index_condition = index[:, -block_size:]
            #getting the predictions
            logits, loss = self.forward(index_condition)
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            #applying softmax funtion to get probability distribution.
            probs = F.softmax(logits, dim=-1) #(B, C)
            #sample from the probability distribution
            index_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            #add the samples index to the running sequence of indices
            index = torch.cat((index,index_next), dim=1) #(B, T+1)
           
        return index

#Un-comment the code below once the model has been trained once and the model-01-pickle file has been created.

model = GPTLanguageModel(vocab_size)
print('loading model parameters....')
with open('model-01-pickle', 'rb') as f: #opens the previously saved parameter files
    model = pickle.load(f) #loads the previous weights/parameters into the model to train further using the loss from previous training.
print('parameters loaded successfully!')
m = model.to(device) #Used to run the model on specified device


#context = torch.zeros((1,1), dtype=torch.long, device=device)
#generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
#print(generated_chars)

#Chatbot code
while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device) #Gets tensor and encode prompt into torch.long data types
    generate_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist()) #decode the chars and unaqeeze. Basically unwrap the matrix
    print(f'Completion:\n[generated_chars]') #printing out the generated characters
