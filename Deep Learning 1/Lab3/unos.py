from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader
import torch
import csv
import numpy as np

@dataclass
class Instance:
    text: list
    label: str

    def __init__(self, text, label):
        self.text = text
        self.label = label


class NPLDataset (Dataset):
    def __init__(self, instances):
        self.instances = instances
        self.text_vocab = None
        self.label_vocab = None
    def __len__(self):
        return len(self.instances)
    def add_vocabs(self, text_vocab, label_vocab):
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def __getitem__(self, index):
        if not self.text_vocab:
            text = self.instances[index].text
            label = self.instances[index].label
        else:
            text = self.text_vocab.encode(self.instances[index].text)
            label = self.label_vocab.encode(self.instances[index].label)
        return text, label
    

class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=0, special=False):
        vocab = sorted(frequencies, key=lambda x: frequencies[x], reverse=True)
        if max_size != -1:
            vocab = vocab[:max_size]
        if min_freq > 0:
            cutoff = -1
            for i in range(len(vocab)):
                if frequencies[vocab[i]] <= min_freq:
                    cutoff = i
                    break
            if cutoff != -1:
                vocab = vocab[:cutoff]

        shift = 0
        self.stoi = {}
        if special:
            shift = 2
            self.stoi = {"<PAD>": 0, "<UNK>":1}

        for i in range(len(vocab)):
            self.stoi[vocab[i]] = i + shift
        
        self.itos = {}
        for token in self.stoi.keys():
            self.itos[self.stoi[token]] = token

    def encode(self, text):
        if type(text) == list:
            enc = []
            for token in text:
                try:
                    enc.append(self.stoi[token])
                except:
                    enc.append(1)
            return torch.tensor(np.array(enc))
    
        else:
            enc = 1
            if text in self.stoi.keys():
                enc = self.stoi[text]
            return torch.tensor(enc)
        


def get_vocabs(dataset, max_size=-1, min_freq=0):
    dict_text = {}
    dict_label = {}
    for text, label in dataset:
        if label not in dict_label.keys():
            dict_label[label] = 1
        else:
            dict_label[label] += 1

        for token in text:
            if token not in dict_text.keys():
                dict_text[token] = 1
            else:
                dict_text[token] += 1
    text_vocab = Vocab(dict_text, max_size, min_freq, True)
    label_vocab = Vocab(dict_label)

    return text_vocab, label_vocab


def get_dataset(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)

        instances = []

        for row in csv_reader:
            text = row[0].split(" ")
            label = row[1].strip(" ")
            instances.append(Instance(text, label))

        dataset = NPLDataset(instances)
    return dataset


def get_vector_reps(filename):
    dict = {}
    with open(filename, 'r') as file:
        for line in file:
            l = line.split(" ")
            word = l[0]
            nums = [float(i) for i in l[1:]]
            dict[word] = np.array(nums)
    return dict

def get_embedding_mat(vocab, filename=None):
    mat = np.random.normal(size=(len(vocab.stoi), 300)).astype("float32")
    mat[0] = np.zeros(300)
    
    if filename:
        vector_dict = get_vector_reps(filename)
        for token, i in vocab.stoi.items():
            if token in vector_dict:
                mat[i] = vector_dict[token]
    
        mat = torch.nn.Embedding.from_pretrained(torch.tensor(mat), True, 0)
    else:
        mat = torch.nn.Embedding.from_pretrained(torch.tensor(mat), False, 0)
    return mat

def pad_collate_fn(batch, pad_index=0):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """
    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts]).to("cuda") # Needed for later
    
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index).to("cuda")
    labels = torch.tensor(labels).to("cuda")
    return texts, labels, lengths





# if __name__=="__main__":
#     train_dataset = get_dataset("sst_train_raw.csv")
#     text_vocab, label_vocab = get_vocabs(train_dataset)
#     train_dataset.add_vocabs(text_vocab, label_vocab)
    
    
#     numericalized_text, numericalized_label = train_dataset[3]
#     # Koristimo nadjačanu metodu indeksiranja
#     print(f"Numericalized text: {numericalized_text}")
#     print(f"Numericalized label: {numericalized_label}")
#     print(len(text_vocab.itos))
#     emb_mat = get_embedding_mat(text_vocab, filename="sst_glove_6b_300d.txt")
#     print(emb_mat.weight)

if __name__=="__main__":

    batch_size = 2 # Only for demonstrative purposes
    shuffle = False # Only for demonstrative purposes
    train_dataset = get_dataset("sst_train_raw.csv")
    text_vocab, label_vocab = get_vocabs(train_dataset)
    train_dataset.add_vocabs(text_vocab, label_vocab)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")

    em = get_embedding_mat(text_vocab)
