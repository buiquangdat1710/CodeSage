from transformers import AutoModel, AutoTokenizer, AutoConfig
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import re
import tensorflow as tf
print("Training LSTM model...")

TEST_SIZE = 0.2
DROP_OUT_P = 0.1
num_epochs = 200
checkpoint = "codesage/codesage-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, add_eos_token=True)
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
config.hidden_size = 512  
model = AutoModel.from_pretrained(checkpoint, config=config, trust_remote_code=True, ignore_mismatched_sizes=True).to(device)

# Load and preprocess data
df = pd.read_csv('full_data.csv')
df  = df[['code', 'label']]
comment_regex = r'(//[^\n]*|\/\*[\s\S]*?\*\/)'
newline_regex = '\n{1,}'
whitespace_regex = '\s{2,}'

def data_cleaning(inp, pat, rep):
    return re.sub(pat, rep, inp)

df['truncated_code'] = (df ['code'].apply(data_cleaning, args=(comment_regex, ''))
                                      .apply(data_cleaning, args=(newline_regex, ' '))
                                      .apply(data_cleaning, args=(whitespace_regex, ' '))
                         )
# remove all data points that have more than 15000 characters
length_check = np.array([len(x) for x in df['truncated_code']]) > 15000
df = df[~length_check]
train_data, valid_data, train_labels, valid_labels = train_test_split(df['code'].values, df['label'].values, test_size=TEST_SIZE, random_state=42)

class CodeDataset(Dataset):
    def __init__(self, data, labels, tokenizer, base_model):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.base_model = base_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx], return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            embedding = self.base_model(**inputs.to(device)).last_hidden_state[:, 0, :].cpu()
        return embedding.squeeze(0), self.labels[idx]

train_dataset = CodeDataset(train_data, train_labels, tokenizer, model)
valid_dataset = CodeDataset(valid_data, valid_labels, tokenizer, model)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, pin_memory=True)

class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers = 10, dropout=0.5):
        super(ImprovedLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_dim, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = nn.functional.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, (hn, cn) = self.lstm(x)
        
        attn_out = self.attention_net(lstm_out, hn[-1])
        
        attn_out = self.batch_norm(attn_out)
        
        out = self.dropout(attn_out)
        out = self.fc(out)
        return out.squeeze()

embedding_dim = 512
hidden_dim = 128
output_dim = 1

model_LSTM = ImprovedLSTMClassifier(embedding_dim, hidden_dim, output_dim, num_layers=10, dropout=0.1).to(device)

criterion = nn.BCEWithLogitsLoss()

# Add L2 regularization
# initial_learning_rate = 0.01
# decay_steps = 10
# decay_rate = 0.9

optimizer = torch.optim.AdamW(model_LSTM.parameters(),lr=0.001, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
scaler = GradScaler()

train_losses = []
val_losses = []

print("Training LSTM model...")

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = (torch.tensor(all_predictions) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_predictions,zero_division=0)
    recall = recall_score(all_labels, all_predictions,zero_division=0)
    f1 = f1_score(all_labels, all_predictions,zero_division=0)

    return avg_loss, accuracy, precision, recall, f1

# Early stopping criteria
patience = 15
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model_LSTM.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with autocast():
            outputs = model_LSTM(inputs)
            loss = criterion(outputs, labels.float())
        predictions = (outputs.squeeze() > 0.5).float()
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)
    accuracy = correct_predictions / total_predictions

    val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model_LSTM, valid_loader, criterion)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, '
          f'Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, '
          f'Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

    torch.cuda.empty_cache()
    # scheduler.step()

    # Early stopping
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     patience_counter = 0
    #     torch.save(model_LSTM.state_dict(), 'best_model.pth')  # Save the best model
    # else:
    #     patience_counter += 1

    # if patience_counter >= patience:
    #     print("Early stopping triggered")
    #     break

# Load the best model
model_LSTM.load_state_dict(torch.load('best_model.pth'))

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss.png')
plt.show()

val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model_LSTM, valid_loader, criterion)

# Calculate ROC AUC
model_LSTM.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_LSTM(inputs)
        probabilities = torch.sigmoid(outputs)
        all_predictions.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

roc_auc = roc_auc_score(all_labels, all_predictions)

print(f'Final Validation Results - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, '
      f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, ROC AUC: {roc_auc:.4f}')
conf_matrix = confusion_matrix(all_labels, (np.array(all_predictions) > 0.5).astype(int))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
