import pandas as pd
import argparse
import wandb
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login()

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'inp_embed_size':{
            'values': [32, 64, 128, 256]
        },
        'dropout': {
            'values': [0.2, 0.3, 0.4]
        },
        'lr': {
            'values': [0.01, 0.001, 0.003]
        },
        'hidden_size': {
            'values': [64, 128, 256]
        },
        'bidirectional': {
            'values': ['Yes','No']
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'cell_type':{
            'values': ['rnn', 'gru', 'lstm']
        }
    }
}

algorithms = {
    'rnn': nn.RNN,
    'gru': nn.GRU,
    'lstm': nn.LSTM
}

sweep_id = wandb.sweep(sweep=sweep_config, project='DL_Assignment3')

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

        self.embedding = nn.Embedding(input_size, config.inp_embed_size)
        self.algo = algorithms[config.cell_type](config.inp_embed_size, config.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.algo(embedded)
        return output, hidden
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, config, output_size):
        super(AttnDecoderRNN, self).__init__()
        self.dropout_p = config.dropout
        hidden_size = config.hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.algo = algorithms[config.cell_type](hidden_size + hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOW_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.algo(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, batch_size, teacher_forcing = True):

    total_loss = 0
    correct = 0
    all_preds=[]
    all_labels=[]
    k = 0

    for data in dataloader:
        input_tensor, target_tensor = data

        target_tensor2 = None
        if (teacher_forcing):
            target_tensor2 = target_tensor

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_outputs, _, attentions = decoder(encoder_outputs, encoder_hidden, target_tensor2)

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
            print(wordFromTensor(input_lang, input_tensor[0]), wordFromTensor(output_lang, target_tensor[0]), wordFromTensor(output_lang, predicted[:45]))
            
    return total_loss / len(dataloader), correct / k

# def show_attention(input_sentence, output_words, attentions):
#     # Convert list of attention weights to a 2D array
#     attentions = np.array(attentions)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     sns.heatmap(attentions[:len(output_words), :len(input_sentence)],
#                 xticklabels=input_sentence, yticklabels=output_words,
#                 cmap='viridis', ax=ax)
#     plt.xlabel('Input Sentence')
#     plt.ylabel('Output Sentence')
#     plt.show()

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
    
    test_loss, test_acc = train_epoch(test_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, config.batch_size, teacher_forcing=False)
    print("Test: accuracy:", test_acc, "Loss: ", test_loss, "\n")

num_epochs = 25


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Parameters')

    parser.add_argument('-wp', '--wandb_project', type=str, default='DL_Assignment3',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-bs', '--batch_size', type= int, default=128, choices = [32, 64, 128], help='Choice of batch size') 
    parser.add_argument('-lr', '--lr', type= float, default=0.001, choices = [0.01, 0.001, 0.003], help='Learning rates')
    parser.add_argument('-ies', '--inp_embed_size', type= int, default=32, choices = [32, 64, 128, 256], help='input embedding size')
    parser.add_argument('-hs', '--hidden_size', type= int, default=256, choices = [64, 128, 256], help='No of neurons in each hidden layer')
    parser.add_argument('-ct', '--cell_type', type= str, default='gru', choices = ['rnn', 'gru', 'lstm'], help='Algorithm / RNN cell type')
    parser.add_argument('-d', '--dropout', type= float, default=0.2, choices = [0.2, 0.3, 0.4], help='Dropout probability')  
    return parser.parse_args()

args = parse_arguments()

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
        'dropout': {
            'values': [args.dropout]
        },
        'lr': {
            'values': [args.lr]
        },
        'hidden_size': {
            'values': [args.hidden_size]
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

def main():
    with wandb.init() as run:
#         wandb.run.name =
        train_dataloader = get_dataloader(x_train, input_lang, output_lang, wandb.config.batch_size)
        val_dataloader = get_dataloader(x_val, input_lang, output_lang, wandb.config.batch_size)
        test_dataloader = get_dataloader(x_test, input_lang, output_lang, wandb.config.batch_size)
        encoder = EncoderRNN(wandb.config, input_lang.n_letters).to(device)
        decoder = AttnDecoderRNN(wandb.config, output_lang.n_letters).to(device)
        print(input_lang.n_letters, output_lang.n_letters)
        train(train_dataloader, val_dataloader, test_dataloader, encoder, decoder, num_epochs, wandb.config)

wandb.agent(sweep_id, function=main, count=1) # calls main function for count number of times.
wandb.finish()