import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, BertForPreTraining, BertPreTrainedModel
from transformers import BertModel, BertConfig, BertTokenizer
from preprocess import sentences, labels  # data after preprocessing [list, list]
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import datetime
import random
import torch.nn as nn
from tqdm import tqdm

class BertClassifer(nn.Module):
    def __init__(self, model, num_classes):
        super(BertClassifer, self).__init__()
        self.bert = model
        self.classifier = nn.Linear(model.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        logits = self.classifier(cls_output)
        return logits




if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='./bert/',  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
)

# sentences = sentences[:10000]
# labels = labels[:10000]
encodings = tokenizer(
    sentences,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors='pt',
    return_attention_mask=True
)
input_ids = encodings['input_ids']
attention_masks = encodings['attention_mask']
uniq_labels = list(set(labels))

print(uniq_labels, len(uniq_labels))
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)
validation_dataloader = DataLoader(
    val_dataset,
    sampler=RandomSampler(val_dataset),
    batch_size=batch_size
)


model = BertModel.from_pretrained(
    pretrained_model_name_or_path='./bert/',
)

class_model = BertClassifer(model=model, num_classes=15)

class_model = class_model.to(device)
optimizer = AdamW(
    class_model.parameters(),
    lr=2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
    eps=1e-8, # args.adam_epsilon  - default is 1e-8.
    no_deprecation_warning=True
)

for epoch in range(2):
    print(f'----------Epoch is {epoch} ----------')
    class_model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch}')
    for i, batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        # print("Batch structure:", [t.shape for t in batch])
        logits = class_model.forward(input_ids, attention_mask)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        if (i + 1) % 100 == 0:
            print(f'Batch {i + 1}: Loss = {loss.item()}')
        loss.backward()
        optimizer.step()
        # print(f"\rBatch loss: {loss.item()}", end='')

class_model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 关闭梯度计算
    total, correct = 0, 0
    for batch in validation_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        logits = class_model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy}')


