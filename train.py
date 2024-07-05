import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertConfig, BertModel, AutoTokenizer

# Constants
TEST_SIZE = 0.2
DROP_OUT_P = 0.1
CHECKPOINT = "codesage/codesage-small"
BATCH_SIZE = 16
NUM_EPOCHS = 200

# Load and preprocess data
df = pd.read_csv('full_data.csv')
train_data = df['code'].values
train_labels = df['label'].values

train_data, valid_data, train_labels, valid_labels = train_test_split(
    train_data, train_labels, test_size=TEST_SIZE, random_state=42)

# Tokenizer and Model Initialization
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True, add_eos_token=True)
config = BertConfig.from_pretrained(CHECKPOINT)
config.hidden_size = 128  # New hidden size
model = BertModel(config)

# Tokenize and encode data
def tokenize_data(data_list):
    data_list = data_list.tolist()
    inputs = tokenizer(data_list, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    return inputs

train_inputs = tokenize_data(train_data)

# Create dataset and dataloader
train_labels_tensor = torch.tensor(train_labels)
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Improved LSTM Classifier
class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(ImprovedLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, hidden_dim, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = nn.functional.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        attn_out = self.attention_net(lstm_out, hn[-1])
        attn_out = self.batch_norm(attn_out)
        out = self.dropout(attn_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Instantiate model, criterion, and optimizer
embedding_dim = 128
hidden_dim = 64
output_dim = 1
model_LSTM = ImprovedLSTMClassifier(embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3)

criterion = nn.BCELoss()
optimizer = optim.Adamax(model_LSTM.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
for epoch in range(NUM_EPOCHS):
    model_LSTM.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()

        # Generate embeddings using the transformer model
        with torch.no_grad():
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask)[0]

        outputs = model_LSTM(embeddings)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        predictions = (outputs.squeeze() > 0.5).float()
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}')

torch.save(model_LSTM.state_dict(), 'model.pth')

# Evaluation function
def evaluate(model_LSTM, model, dataloader):
    model_LSTM.eval()
    model.eval()
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask)[0]

            outputs = model_LSTM(embeddings)
            loss = criterion(outputs.squeeze(), labels.float())
            epoch_loss += loss.item()

            predictions = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_predictions
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1, epoch_loss / len(dataloader)

# Create validation dataloader
valid_inputs = tokenize_data(valid_data)
valid_labels_tensor = torch.tensor(valid_labels)
valid_dataset = TensorDataset(valid_inputs['input_ids'], valid_inputs['attention_mask'], valid_labels_tensor)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluate the model
accuracy, precision, recall, f1, val_loss = evaluate(model_LSTM, model, valid_loader)

print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')
print(f'Validation Precision: {precision:.4f}')
print(f'Validation Recall: {recall:.4f}')
print(f'Validation F1 Score: {f1:.4f}')
