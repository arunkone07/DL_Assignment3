import pandas as pd
import argparse
import wandb
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login()
# 3dc8367198d0460ba99efb94e713de7e299e685d

algorithms = {
    'rnn': nn.RNN,
    'gru': nn.GRU,
    'lstm': nn.LSTM
}

SOW_token = 0
EOW_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.letter2index = {}
        self.letter2count = {}
        self.index2letter = {0: "0", 1: "1"}
        self.n_letters = 2 # Count SOW and EOW

    def addWord(self, word):
        for ch in word:
            self.addLetter(ch)

    def addLetter(self, ch):
        if ch not in self.letter2index:
            self.letter2index[ch] = self.n_letters
            self.letter2count[ch] = 1
            self.index2letter[self.n_letters] = ch
            self.n_letters += 1
        else:
            self.letter2count[ch] += 1

input_lang = Lang('eng')
output_lang = Lang('hin')


x_train = pd.read_csv('./aksharantar_sampled/hin/hin_train.csv', header=None) #, nrows=1000)
x_val = pd.read_csv('./aksharantar_sampled/hin/hin_valid.csv', header=None)
x_test = pd.read_csv('./aksharantar_sampled/hin/hin_test.csv', header=None)
sz = x_train[0]

MAX_LENGTH = 50

def indexesFromWord(lang, word):
    return [lang.letter2index[ch] for ch in word]

def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOW_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def wordFromTensor(lang, tensor):
    s = ""
    for i in tensor:
        if(i.item()==1):
            break
        s += lang.index2letter[i.item()] 
    return s

def get_dataloader(x, input_lang, output_lang, batch_size):
    n = len(x[0])
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for i in range(n):
        input_lang.addWord(x[0][i])
        output_lang.addWord(x[1][i])
        inp_ids = indexesFromWord(input_lang, x[0][i])
        tgt_ids = indexesFromWord(output_lang, x[1][i])
        inp_ids.append(EOW_token)
        tgt_ids.append(EOW_token)
        input_ids[i, :len(inp_ids)] = inp_ids
        target_ids[i, :len(tgt_ids)] = tgt_ids
    
    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

class EncoderRNN(nn.Module):
    def __init__(self, config, input_size):
        super(EncoderRNN, self).__init__()
        
        self.bidirectional = False
        if(config.bidirectional == 'Yes'):
            self.bidirectional = True
            
        self.embedding = nn.Embedding(input_size, config.inp_embed_size)
        self.algo = algorithms[config.cell_type](config.inp_embed_size, config.hidden_size, config.num_layers, bidirectional = self.bidirectional, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.algo(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, config, output_size):
        super(DecoderRNN, self).__init__()
        
        self.config = config
        self.bidirectional = False
        if(config.bidirectional == 'Yes'): 
            self.bidirectional = True
            
        self.embedding = nn.Embedding(output_size, config.hidden_size)
        self.algo = algorithms[config.cell_type](config.hidden_size, config.hidden_size, config.num_layers, bidirectional = self.bidirectional, batch_first=True)
        self.out = nn.Linear(config.hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOW_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = F.relu(self.embedding(input))
        output, hidden = self.algo(output, hidden)
        output = self.out(output)
        return output, hidden

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, batch_size, teacher_forcing = True):

    total_loss = 0
    correct = 0
    k = 0
    
    for data in dataloader:
        input_tensor, target_tensor = data
        
        target_tensor2 = None
        if (teacher_forcing):
            target_tensor2 = target_tensor
            
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
      
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor2)
        
        outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
        labels = target_tensor.view(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        
        i = 0
        while (i < batch_size * MAX_LENGTH):
            j = 0
            while (j < MAX_LENGTH):
                if(predicted[i+j] != labels[i+j]):
                    break
                j+=1
            if(j==MAX_LENGTH):
                correct += 1
            i += MAX_LENGTH
        k += batch_size
        
        if(k%6400==0):
            print(k, loss.item(), correct)
#             print(wordFromTensor(input_lang, input_tensor[0]), wordFromTensor(output_lang, target_tensor[0]), wordFromTensor(output_lang, predicted[:45]))
        
    return total_loss / len(dataloader), correct / k

def train(train_dataloader, val_dataloader, test_dataloader, encoder, decoder, n_epochs, config):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        print(epoch)
        loss, acc = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, config.batch_size)
        print("Train: accuracy:", acc, "loss:", loss)
        if(acc<0.01 and epoch>=15):
            break
        wandb.log({'train_accuracy': acc})
        wandb.log({'train_loss': loss})
        val_loss, val_acc = train_epoch(val_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, config.batch_size, teacher_forcing=False)
        print("Validation: accuracy:", val_acc, "Loss:", val_loss, "\n")
        wandb.log({'val_accuracy': val_acc})
        wandb.log({'val_loss': val_loss})
    loss, acc = train_epoch(test_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, config.batch_size, teacher_forcing=False)
    print("Test: accuracy:", acc, "Loss:", loss, "\n")

def evaluate(encoder, decoder):
    for i in range(10):
        print('>', x_test[0][i])
        print('=', x_test[1][i])
        output = ''
        
        with torch.no_grad():
            input_tensor = tensorFromWord(input_lang, x_test[0][i])

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_word = ''
            for idx in decoded_ids:
                if idx.item() == EOW_token:
                    decoded_word+='1'
                    break
                decoded_word += output_lang.index2letter[idx.item()]
            print('<', decoded_word)
                  

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')

    parser.add_argument('-wp', '--wandb_project', type=str, default='DL_Assignment3',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-bs', '--batch_size', type= int, default=64, choices = [32, 64, 128], help='Choice of batch size') 
    parser.add_argument('-lr', '--lr', type= float, default=0.001, choices = [0.01, 0.001, 0.003], help='Learning rates')
    parser.add_argument('-ies', '--inp_embed_size', type= int, default=256, choices = [32, 64, 128, 256], help='input embedding size')
    parser.add_argument('-hs', '--hidden_size', type= int, default=256, choices = [64, 128, 256], help='No of neurons in each hidden layer')
    parser.add_argument('-nl', '--num_layers', type= int, default=3, choices = [1, 2, 3], help='No of layers in encoder and decoder')
    # parser.add_argument('-dl', '--dec_layers', type= int, default=, choices = [1, 2, 3], help='No of layers in decoder')
    parser.add_argument('-bd', '--bidirectional', type= str, default='No', choices = ['Yes', 'No'], help='Bidirectional RNN or not')
    parser.add_argument('-ct', '--cell_type', type= str, default='lstm', choices = ['rnn', 'gru', 'lstm'], help='Algorithm / RNN cell type')
    parser.add_argument('-d', '--dropout', type= float, default=0.4, choices = [0.2, 0.3, 0.4], help='Dropout probability')  
    return parser.parse_args()

args = parse_arguments()

# sweep_id = wandb.sweep(sweep=sweep_config, project='DL_Assignment3')
# wandb.init(project=args.wandb_project)


best_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'inp_embed_size':{
            'values': [args.inp_embed_size]
        },
        'num_layers': {
            'values': [args.num_layers]
        },
        'dropout': {
            'values': [args.dropout]
        },
        'lr': {
            'values': [args.lr]
        },
        'hidden_size': {
            'values': [args.hidden_size]
        },
        'bidirectional': {
            'values': [args.bidirectional]
        },
        'batch_size': {
            'values': [args.batch_size]
        },
        'cell_type':{
            'values': [args.cell_type]
        }
    }
}

sweep_id = wandb.sweep(sweep=best_config, project='DL_Ass3')

num_epochs = 25

def main():
    with wandb.init() as run:
        train_dataloader = get_dataloader(x_train, input_lang, output_lang, wandb.config.batch_size)
        val_dataloader = get_dataloader(x_val, input_lang, output_lang, wandb.config.batch_size)
        test_dataloader = get_dataloader(x_test, input_lang, output_lang, wandb.config.batch_size)
        encoder = EncoderRNN(wandb.config, input_lang.n_letters).to(device)
        decoder = DecoderRNN(wandb.config, output_lang.n_letters).to(device)
        print(input_lang.n_letters, output_lang.n_letters)
        train(train_dataloader, val_dataloader, test_dataloader, encoder, decoder, num_epochs, wandb.config)
        encoder.eval()
        decoder.eval()
        evaluate(encoder, decoder)


wandb.agent(sweep_id, function=main, count=1)
wandb.finish()