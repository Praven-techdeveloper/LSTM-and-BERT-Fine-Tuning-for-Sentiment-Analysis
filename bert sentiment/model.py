import torch
import torch.nn as nn
from transformers import BertModel

class BertLSTM(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', lstm_hidden_size=256, num_classes=2):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden_size * 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_outputs.last_hidden_state
        
        # LSTM processing
        lstm_output, _ = self.lstm(sequence_output)
        
        # Extract last hidden state (combined forward/backward)
        last_forward = lstm_output[:, -1, :self.lstm.hidden_size]
        last_backward = lstm_output[:, 0, self.lstm.hidden_size:]
        combined = torch.cat((last_forward, last_backward), dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return logits