import torch
import torch.nn as nn
import unos
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class BaselineModel(nn.Module):
    def __init__(self, embedding_matrix, input_size=300, hidden_size=150):
        super(BaselineModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.mean = torch.mean
        self.em = embedding_matrix
        
    def forward(self, x):
        x = self.em(x)
        x = self.mean(x, dim=1, dtype=torch.float32)
        x = x.squeeze()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze()

        return x


def train(model, data, optimizer, criterion, clip):
  model.train()
  for i, (batch_x, batch_y, lens) in enumerate(data):
    model.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y.float())
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
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
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    print("F1:", f1)
    print("Confusion matrix:\n", confusion_mat)



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
    epochs = 5
    batch_size_train = 10
    batch_size_test = 32
    clip = 0.25
    print("SEED EPOCHS BS_TRAIN BS_TEST CLIP")
    print(seed, epochs, batch_size_train, batch_size_test, clip)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, valid_dataset, test_dataset, embedding_matrix = load_dataset(-1, 1)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, 
            shuffle=True, collate_fn=unos.pad_collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, 
            shuffle=True, collate_fn=unos.pad_collate_fn)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size_test, 
            shuffle=True, collate_fn=unos.pad_collate_fn)
    model = BaselineModel(embedding_matrix)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    accs = []
    f1s = []
    mats = []
    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, clip)
        #print("Epoch", epoch)
        loss, accuracy, f1, confusion_mat = evaluate(model, valid_loader, criterion)
        losses += [loss]
        accs += [accuracy]
        f1s += [f1]
        mats += [confusion_mat]

    print("Loss:")
    for i in losses:
        print("\t", i)
    
    print("Accuracy:")
    for acc in accs:
        print("\t", acc)

    print("F1 Score:")
    for f1_score in f1s:
        print("\t", f1_score)

    print("Confusion Matrix:")
    for mat in mats:
        print(mat)

    print("Test:")
    loss, accuracy, f1, confusion_mat = evaluate(model, test_loader, criterion)

    print("Loss:", loss)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_mat, sep="")

if __name__=="__main__":
    main()
    
