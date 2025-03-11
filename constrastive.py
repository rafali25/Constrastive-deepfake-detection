#LeakyRelu Activation Function 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sklearn.metrics
from tqdm import tqdm

# Define the Generator class with increased capacity and BatchNorm
class Generator(nn.Module):
    def _init_(self, input_size=1280, hidden_size=2000, dropout_prob=0.2):
        super(Generator, self)._init_()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        return self.lrelu3(x)

# Define the ContrastiveModel class with Leaky ReLU activation
class ContrastiveModel(nn.Module):
    def _init_(self, h=1280, dropout_prob=0.2):
        super(ContrastiveModel, self)._init_()
        self.generator = Generator(input_size=h, hidden_size=2000, dropout_prob=dropout_prob)
        self.fc = nn.Linear(2000, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        features = self.generator(x)
        return self.fc(self.lrelu(features))

# Define the NumpyDeepfakeDetectionDataset class
class NumpyDeepfakeDetectionDataset(Dataset):
    def _init_(self, features_file, labels_file=None, target_size=1280):
        self.features = np.load(features_file)
        self.labels = np.load(labels_file) if labels_file else None
        self.target_size = target_size

    def _len_(self):
        return len(self.features)

    def _getitem_(self, idx):
        feature = self.features[idx]
        if feature.shape[0] > self.target_size:
            feature = feature[:self.target_size]
        elif feature.shape[0] < self.target_size:
            padding = np.zeros((self.target_size - feature.shape[0],))
            feature = np.concatenate((feature, padding))
        feature = torch.tensor(feature, dtype=torch.float32)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return feature, label
        return feature

# Define the compute_eer function
def compute_eer(label, pred, positive_label=1):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_1 + eer_2) / 2
    return eer

# Main training loop
def main(args):
    # Load datasets
    train_set = NumpyDeepfakeDetectionDataset(args['train_features_file'], args['train_labels_file'])
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
    
    val_set = NumpyDeepfakeDetectionDataset(args['val_features_file'], args['val_labels_file'])
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

    model = ContrastiveModel(h=args['h'], dropout_prob=args['dropout_prob'])
    model.to(args['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    for epoch in range(args['num_epochs']):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{args['num_epochs']}]"):
            features, labels = batch
            features, labels = features.to(args['device']), labels.to(args['device'])
            optimizer.zero_grad()

            output = model(features)
            output = output.squeeze()
            labels = labels.squeeze()

            loss = nn.BCEWithLogitsLoss()(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args['num_epochs']}], Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        all_val_labels = []
        all_val_outputs = []
        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch
                features = features.to(args['device'])
                output = model(features)
                all_val_labels.append(labels.numpy())
                all_val_outputs.append(output.cpu().numpy())

        all_val_labels = np.concatenate(all_val_labels)
        all_val_outputs = np.concatenate(all_val_outputs)
        val_eer = compute_eer(all_val_labels, all_val_outputs)
        print(f"Epoch [{epoch + 1}/{args['num_epochs']}], Validation EER: {val_eer:.4f}")

        # Step the learning rate scheduler based on validation loss
        lr_scheduler.step(avg_loss)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        if 'test_features_file' in args and 'test_labels_file' in args:
            test_set = NumpyDeepfakeDetectionDataset(args['test_features_file'], args['test_labels_file'])
            test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

            test_labels = []
            test_outputs = []
            for batch in test_loader:
                features, labels = batch
                features = features.to(args['device'])
                output = model(features)
                test_labels.append(labels.numpy())
                test_outputs.append(output.cpu().numpy())

            test_labels = np.concatenate(test_labels)
            test_outputs = np.concatenate(test_outputs)
            test_eer = compute_eer(test_labels, test_outputs)
            print(f"Test Equal Error Rate (Test EER): {test_eer:.4f}")
            print(f"Test Equal Error Rate in percentage(Test EER): {test_eer*100:.4f}")

if _name_ == "_main_":
    args = {
        'train_features_file': '/kaggle/input/decro-english/DECRO_ENGLISH/MMS_NEW/train/mms_train_features.npy',
        'train_labels_file': '/kaggle/input/decro-english/DECRO_ENGLISH/MMS_NEW/train/mms_train_labels.npy',
        'val_features_file': '/kaggle/input/decro-english/DECRO_ENGLISH/MMS_NEW/val/mms_val_features.npy',
        'val_labels_file': '/kaggle/input/decro-english/DECRO_ENGLISH/MMS_NEW/val/mms_val_labels.npy',
        'test_features_file': '/kaggle/input/decro-english/DECRO_ENGLISH/MMS_NEW/test/mms_test_features.npy',
        'test_labels_file': '/kaggle/input/decro-english/DECRO_ENGLISH/MMS_NEW/test/mms_test_labels.npy',
        'batch_size': 32,
        'num_epochs': 25,
        'lr': 1e-5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'h': 1280,
        'dropout_prob': 0.2,
        'weight_decay': 5e-6
    }

    main(args)
