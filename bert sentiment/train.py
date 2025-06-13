import torch
import torch.nn as nn  # IMPORT MISSING HERE
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from datasets import load_dataset
from model import BertLSTM
from tqdm import tqdm

# Configuration
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
MAX_LEN = 256
EPOCHS = 4
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare dataset
def prepare_data():
    dataset = load_dataset('imdb')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )
    
    dataset = dataset.map(tokenize, batched=True, batch_size=1000)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset, tokenizer

# Training function
def train():
    # Prepare data
    dataset, tokenizer = prepare_data()
    train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE)
    
    # Initialize model
    model = BertLSTM().to(DEVICE)
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': LR},
        {'params': model.lstm.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    criterion = nn.CrossEntropyLoss()  # NOW PROPERLY DEFINED
    
    # Training loop
    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE)
            }
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(DEVICE),
                    'attention_mask': batch['attention_mask'].to(DEVICE)
                }
                labels = batch['label'].to(DEVICE)
                
                outputs = model(**inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'models/bert_lstm_model.pt')
            tokenizer.save_pretrained('models/tokenizer')
    
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    train()