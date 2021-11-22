#import library
import pandas as pd
from tensorflow import keras
from keras import layers
import re
from IPython import embed
import torch
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from torch import nn
import math


#initialize list of stop words
def init_stop_words():
    stop_words_string = 'bị, bởi, cả, các, cái, cần, càng, chỉ, chiếc, cho, chứ, chưa, chuyện, có, có_thể, cứ, của, cùng, cũng, đã, đang, đây, để, đến_nỗi, đều, điều, do, đó, được, dưới, gì, khi, không, là, lại, lên, lúc, mà, mỗ, một_các, nà, nên, nếu, ngay, nhiều, như, nhưng, những, nơi, nữa, phải, qua, ra, rằng, rằng, rất, rất, rồi, sau, sẽ, tại, theo, thì, trên, trước, từ, từng, và, vẫn, vào, vậy, vì, việc, với, vừa'
    stop_words = re.split(r'(\W|[0-9\s])+', stop_words_string)
    remove_token = re.compile(r'(\W|[0-9\s])+')
    i = 0
    while i < len(stop_words):
        if(remove_token.fullmatch(stop_words[i]) != None):
            stop_words.pop(i)
            i = i - 1
        i = i + 1
    return stop_words

_stop_words = init_stop_words()



#initialize transformer class
class self_transformer(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(self_transformer, self).__init__()
        
            self.linear1 = nn.Linear(input_dim, output_dim)
            self.linear2 = nn.Linear(input_dim, output_dim)
            self.linear3 = nn.Linear(input_dim, output_dim)
            self.softmax = nn.Softmax(-1)
        def forward(self, _input, output_dim):
            question = self.linear1(_input) # 30x128
            key = self.linear2(_input) # 30x128
            value = self.linear3(_input) # 30x128
            self_attention_ = torch.mm(key, torch.transpose(question, 0, 1))/math.sqrt(output_dim)# 30x30
            self_attention = self.softmax(self_attention_) # 30x30
            self_output = value.unsqueeze(0)*self_attention.unsqueeze(-1)
            return self_output.sum(1)

class multi_head_transformer(nn.Module):
    def __init__(self, input_dim, output_dim, max_length, num_class):
        super(multi_head_transformer, self).__init__()
        self.linear = []
        for i in range(num_class):
            self.linear.append(nn.Linear(input_dim, output_dim))
        self.linear_1 = nn.Linear(output_dim*max_length, 7)
    def forward(self, multi,output_dim):
        result = torch.zeros((multi[0].shape[0], output_dim), dtype=torch.float32)
        for i in range(len(multi)):
            result += self.linear[i](multi[i])
        result = torch.flatten(result)
        result = self.linear_1(result)
        result = torch.tanh(result)
        return result


def transform(_input, model_1, model_2, output_dim_encode,output_dim_decode):
    encoders = []
    for i in range(0, 8):
        encoders.append(model_1[i](_input, output_dim_encode))
    decoder = model_2(encoders, output_dim_decode)
    return decoder


data = pd.read_csv('mebe_shopee.csv')
print(data.head())
sentences = []
label_list = []
for i in range(len(data)):
    sentences.append(data['cmt'].loc[i])
for i in range(len(data)):
    label = []
    label.append(float(data['giá'].loc[i]))
    label.append(float(data['dịch_vụ'].loc[i]))
    label.append(float(data['an_toàn'].loc[i]))
    label.append(float(data['chất_lượng'].loc[i]))
    label.append(float(data['ship'].loc[i]))
    label.append(float(data['other'].loc[i]))
    label.append(float(data['chính_hãng'].loc[i]))
    label_list.append(label)





train_text = []
train_label = []
for i in range(int(4*len(sentences)/5)):
    train_text.append(str(sentences[i]))
    train_label.append(label_list[i])

test_text = []
test_label = []
for i in range(int(4*len(sentences)/5), len(sentences)):
    test_text.append(str(sentences[i]))
    test_label.append(label_list[i])


#lowering ang spliting
for i in range(0, len(train_text)):
    train_text[i] = train_text[i].lower()

for i in range(0, len(test_text)):
    test_text[i] = test_text[i].lower()

train_text_process = []
test_text_process = []
for sen in train_text:
    train_text_process.append(re.split(r'(\W+|[0-9\s])+', sen))
for sen in test_text:
    test_text_process.append(re.split(r'(\W+|[0-9\s])+', sen))
    
    
    
    
    
#remove puctuation, icon, strange character
train_text = train_text_process
test_text = test_text_process

remove_token = re.compile(r'(\W+|[0-9\s])+')

for i in range(0, len(train_text)):
    j = 0
    while(j < len(train_text[i])):
        if remove_token.fullmatch(train_text[i][j]) != None or train_text[i][j] == '':
            train_text[i].pop(j)
            j = j - 1
        j = j + 1

for i in range(0, len(test_text)):
    j = 0
    while(j < len(test_text[i])):
        if remove_token.fullmatch(test_text[i][j]) != None or test_text[i][j] == '':
            test_text[i].pop(j)
            j = j - 1
        j = j + 1

print(train_text[0:100])



#make a list of word from orginal data
list_words = []

for sen in train_text:
    list_words.extend(sen)
for sen in test_text:
    list_words.extend(sen)
    
    

#initialize a dictionary of vocabualry also remove stop words by putting them into OOV
list_vocab = sorted(set(list_words))
print(len(list_words))
dict_word = {}

for word in list_words:
    dict_word[word] = 0

for word in list_words:
    dict_word[word] = dict_word[word] + 1
# for word in dict_word.keys():
#     if word in _stop_words:
#         dict_word[word] = 0
print(len(dict_word))
sorted_keys = sorted(dict_word, key=dict_word.get)



#initialize some parameter for padding
vocab_size = len(dict_word) - 100
embed_dim = 128
max_length = 30
trunc_type = 'post'
pad_type = 'pre'
oov_tok = 'OOV'
batch_size = 8

#take words to dictionary of vocabulary refer to some priorities
dict_vocab = {}
index = 1
for i in range(len(sorted_keys) - 1, len(sorted_keys) - vocab_size, -1):
    dict_vocab[sorted_keys[i]] = index
    index = index + 1
dict_vocab['OOV'] = index
for i in range(0, 100):
    print(dict_word[sorted_keys[i]])



#change word in sentence to an integer value
min_fre = dict_word[sorted_keys[len(sorted_keys)-vocab_size]]
print(min_fre)
print('shop' in dict_word.keys())
sen_to_seq_train = []
for sen in train_text:
    list_seq = []
    for word in sen:
        if dict_word[word] > min_fre:
            print(word)
            list_seq.append(dict_vocab[word])
        else:
            if dict_word[word] < min_fre:
                list_seq.append(dict_vocab['OOV'])
            else:
                if word in dict_vocab.keys():
                    list_seq.append(dict_vocab[word])
                else:
                    list_seq.append(dict_vocab['OOV'])
    sen_to_seq_train.append(list_seq)

sen_to_seq_test = []
for sen in test_text:
    list_seq = []
    for word in sen:
        if not word in dict_word.keys():
            list_seq.append(dict_vocab['OOV'])
        else:
            if dict_word[word] > min_fre:
                list_seq.append(dict_vocab[word])
            else:
                if dict_word[word] < min_fre:
                    list_seq.append(dict_vocab['OOV'])
                else:
                        if word in dict_vocab.keys():
                            list_seq.append(dict_vocab[word])
                        else:
                            list_seq.append(dict_vocab['OOV'])
    sen_to_seq_test.append(list_seq)
    
    
    
#padding
train_seq = np.array(sen_to_seq_train)

train_pad = pad_sequences(train_seq,maxlen=max_length, padding=pad_type, truncating=trunc_type)
test_seq = np.array(sen_to_seq_test)
test_pad = pad_sequences(test_seq,maxlen=max_length, padding=pad_type, truncating=trunc_type)
print(train_pad[0].shape)
print(train_label)
train_pad = torch.from_numpy(train_pad)
test_pad = torch.from_numpy(test_pad)
train_label = torch.tensor(train_label, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype=torch.float32)
print(test_label.shape)



#process pads into input before putting in transformer models

model_1 = []
for i in range(8):
    model_1.append(self_transformer(input_dim=embed_dim, output_dim = 64))
model_2 = multi_head_transformer(input_dim=64, output_dim=embed_dim, max_length=max_length, num_class=8)
embed_model = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim=embed_dim)
input_train = []
for i in range(train_pad.shape[0]):
    input_train.append(embed_model(train_pad[i]))
model_2.train()
for p in model_1:
  p.train()
print(input_train[0])
input_test = []
for i in range(test_pad.shape[0]):
    input_test.append(embed_model(test_pad[i]))

position_encode = torch.zeros((max_length, embed_dim))
for i in range(max_length):
    for j in range(embed_dim):
        if j % 2 == 1:
            position_encode[i][j] = math.sin(float(i+1)/10000**(float(j+1)/embed_dim))
        else:
            position_encode[i][j] = math.cos(float(i+1)/10000**(float(j)/embed_dim))
for i in range(len(input_train)):
      input_train[i] += position_encode
for i in range(len(input_test)):
      input_test[i] += position_encode
# for i in range(len(input_test)):
#     print(input_test[i].shape)
criterion = nn.L1Loss()



optimizer = torch.optim.Adam([{'params':model_1[0].parameters()}, 
                              {'params':model_1[1].parameters()}, 
                              {'params':model_1[2].parameters()}, 
                              {'params':model_1[3].parameters()}, 
                              {'params':model_1[4].parameters()}, 
                              {'params':model_1[5].parameters()}, 
                              {'params':model_1[6].parameters()}, 
                              {'params':model_1[7].parameters()}, 
                              {'params':model_2.parameters()}], lr=0.000105)

#traing and testing
def train(model_1, model_2, input_train, train_label, input_test, test_label, batch_size, optimizer, criterion, epochs = 10):
    useful_stuff = {'training_loss':[], 'validation_accuracy':[]}
    for epoch in range(epochs):
        num_batch = int((len(input_train)-1)/batch_size)
        correct_train = 0
        for i in range(num_batch + 1):
            optimizer.zero_grad()
            loss = torch.tensor(0.0)
            for j in range(i*batch_size, min([(i+1)*batch_size,len(input_train)])):
                y = transform(input_train[j], model_1, model_2, 64,embed_dim)
                
                print(y.mean())
                count = 0
                
                for k in range(0, y.shape[0]):
                    if abs(y[k].data - train_label[j][k].data) > 0.5:
                        count += 1;
                if count < 2: correct_train += 1
            
                # if abs(y.mean().data - train_label[j].mean().data) < 0.5:
                #     correct_train += 1 
                print(float(correct_train)/(j + 1 + i * batch_size))
                
                loss += criterion(y.unsqueeze(0), train_label[j])
                # print(loss.requires_grad)
            loss.backward(retain_graph=True)
            optimizer.step()    
            useful_stuff['training_loss'].append(loss.data)
            print(loss.data)
        for i in range(100):
            print('complete')
        correct = 0.0
        t = 0.5
        for i in range(len(input_test)):
            y = transform(input_test[i], model_1, model_2, 64,embed_dim)
            print(y.shape)
            count = 0.0
            for k in range(0, y.shape[0]):
                if abs(y[k].data - test_label[i][k].data) > 0.5:
                    correct += 1;
            # if count < 2: correct += 1
      
            # if abs(y.mean().data - test_label[i].mean().data) < 0.5:
            #     correct += 1 
        accuracy = correct/test_label.shape[0]/7
        useful_stuff['validation_accuracy'].append(accuracy)
    return useful_stuff

def predict(input):
    input1 = input.lower()
    input1 = re.split(r'(\W+|[0-9\s])+', input1)
    remove_token = re.compile(r'(\W+|[0-9\s])+')
    j = 0
    while (j < len(input1)):
        if remove_token.fullmatch(input1[j]) != None or input1[j] == '':
            input1.pop(j)
            j = j - 1
        j = j + 1

    print(input1)

    seq = []
    for word in input1:
        if not word in dict_word.keys():
            seq.append(dict_vocab['OOV'])
            print('No')
            continue
        if dict_word[word] > min_fre:
            seq.append(dict_vocab[word])
        else:
            if dict_word[word] < min_fre:
                seq.append(dict_vocab['OOV'])
            else:
                if word in dict_vocab.keys():
                    seq.append(dict_vocab[word])
                else:
                    seq.append(dict_vocab['OOV'])
    if len(seq) == 0:
        seq.append(0)
    
    
    seq_ = []
    seq_.append(seq)
    seq = np.array(seq_)
    print(seq)
    # train_pad = pad_sequences(train_seq,maxlen=max_length, padding=pad_type, truncating=trunc_type)
    pad = pad_sequences(seq, maxlen=max_length, padding=pad_type, truncating=trunc_type)
    pad = torch.from_numpy(pad)
    pad = embed_model(pad)

    y = transform(pad[0], model_1, model_2, 64, embed_dim)
    result = []
    print(y)
    for k in range(0, y.shape[0]):
        if y[k].data > 0.5:
            result.append(1)
        else: 
            if y[k].data < -0.5:
                result.append(-1)
            else:
                result.append(0)
    return result






