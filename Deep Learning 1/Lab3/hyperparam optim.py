import torch
import torch.nn as nn
import unos
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys

device = "cuda"

class RNNModel(nn.Module):
    def __init__(self, embedding_matrix, rnn_type="RNN", dropout = 0, bi=False, num_layers=2, input_size=300, hidden_size=150):
        super(RNNModel, self).__init__()
        if rnn_type == "RNN":
            self.rnns = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bi).to(device)
        elif rnn_type == "GRU":
            self.rnns = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bi).to(device)
        elif rnn_type == "LSTM":
            self.rnns = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bi).to(device)
        if bi:
            self.fc1 = nn.Linear(2*hidden_size, 150).to(device)
        else:
            self.fc1 = nn.Linear(hidden_size, 150).to(device)
        self.relu = nn.ReLU().to(device)
        self.fc2 = nn.Linear(150, 1).to(device)
        self.em = embedding_matrix.to(device)


    def forward(self, x):
        x = self.em(x)
        x = x.transpose(0, 1)
        x = x.float()
        x, _ = self.rnns(x)
        x = self.fc1(x[-1])
        x = self.relu(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x



def train(model, data, optimizer, criterion, clip):
  model.train()
  for i, (batch_x, batch_y, lens) in enumerate(data):
    model.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()


def evaluate(model, data, criterion):
    losses = []
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, lens) in enumerate(data):
            logits = model(batch_x)
            loss = criterion(logits, batch_y.float())

            losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            predictions.extend(preds.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    loss = sum(losses) / len(losses)
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    confusion_mat = confusion_matrix(targets, predictions)

    return loss, accuracy, f1, confusion_mat




def load_dataset(max_size, min_freq):
    train_dataset = unos.get_dataset("sst_train_raw.csv")
    text_vocab, label_vocab = unos.get_vocabs(train_dataset, max_size, min_freq)
    train_dataset.add_vocabs(text_vocab, label_vocab)
    emb_mat = unos.get_embedding_mat(text_vocab, "sst_glove_6b_300d.txt")

    test_dataset = unos.get_dataset("sst_test_raw.csv")
    test_dataset.add_vocabs(text_vocab, label_vocab)

    valid_dataset = unos.get_dataset("sst_valid_raw.csv")
    valid_dataset.add_vocabs(text_vocab, label_vocab)

    return train_dataset, valid_dataset, test_dataset, emb_mat

def main():
    seed = 7052020
    epochs = 10
    batch_size_test = 32
    max_size = -1
    print("SEED EPOCHS BS_TEST MAX_SIZE")
    print(seed, epochs, batch_size_test, max_size)
    np.random.seed(seed)
    torch.manual_seed(seed)


    
    
    criterion = nn.BCEWithLogitsLoss()

    for min_freq in [1, 10, 100]:
        train_dataset, _, test_dataset, embedding_matrix = load_dataset(max_size, min_freq)
        for batch_size_train in [10, 20, 30]:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, 
                shuffle=True, collate_fn=unos.pad_collate_fn)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, 
                shuffle=True, collate_fn=unos.pad_collate_fn)
            for clip in [0.1, 0.25, 0.5]:
                for num_layers in [2, 3, 4]:
                    for dropout in [0, 0.2, 0.5]:
                        print("MIN_FREQ BATCH_SIZE CLIP NUM_LAYERS DROPOUT")
                        print(min_freq, batch_size_train, clip, num_layers, dropout)
                        model = RNNModel(embedding_matrix, rnn_type="GRU", dropout=dropout, num_layers=num_layers)
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                        for epoch in range(epochs):
                            train(model, train_loader, optimizer, criterion, clip)

                        print("Test:")
                        loss, accuracy, f1, confusion_mat = evaluate(model, test_loader, criterion)

                        print("Loss:", loss)
                        print("Accuracy:", accuracy)
                        print("F1 Score:", f1)
                        print("Confusion Matrix:\n", confusion_mat, sep="")

    

if __name__=="__main__":
    main()
    
