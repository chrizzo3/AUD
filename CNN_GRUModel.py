import torch.nn as nn
import torch.nn.functional as F


class CNNGRUModel(nn.Module):
    def __init__(self, 
                 input_channels=9, 
                 num_joints=25, 
                 cnn_out_channels=64, 
                 gru_hidden_size=128, 
                 num_classes=15):
        super(CNNGRUModel, self).__init__()
        
        # CNN per l'estrazione delle caratteristiche spaziali
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn_out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout per regolarizzazione
            nn.Conv2d(cnn_out_channels, cnn_out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU()
        )
        
        # GRU per catturare la dinamica temporale
        self.gru = nn.GRU(
            input_size=cnn_out_channels * num_joints,
            hidden_size=gru_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,  # Dropout tra i layer
            bidirectional=True
        )
        
        # Fully connected per la classificazione
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, gru_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout finale
            nn.Linear(gru_hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, num_bodies, num_joints, feature_size)
        
        # Unisci i corpi e joint per passare nella CNN
        batch_size, seq_len, num_bodies, num_joints, feature_size = x.size()
        x = x.view(-1, num_bodies * feature_size, num_joints, seq_len)  # (batch_size * num_bodies, input_channels, num_joints, seq_len)
        x = self.cnn(x)  # (batch_size * num_bodies, cnn_out_channels, num_joints, seq_len)
        
        # Ridimensiona per la GRU
        x = x.permute(0, 3, 1, 2)  # (batch_size * num_bodies, seq_len, cnn_out_channels, num_joints)
        x = x.contiguous().view(batch_size, seq_len, -1)  # (batch_size, seq_len, cnn_out_channels * num_joints)
        
        # Passa nella GRU
        x, _ = self.gru(x)  # (batch_size, seq_len, gru_hidden_size * 2)
        
        # Considera solo l'ultimo output della sequenza
        x = x[:, -1, :]  # (batch_size, gru_hidden_size * 2)
        
        # Classificazione finale
        x = self.fc(x)  # (batch_size, num_classes)
        return x
