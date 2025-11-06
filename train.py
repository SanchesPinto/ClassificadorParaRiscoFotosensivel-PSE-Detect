from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
# import cv2
import numpy as np
import random 
import torchvision.models as models
#import pandas as pd

# ============================
# 1. Transformações e Dataset 
# ============================

IMG_SIZE = 224 
transform = transforms.Compose([
    transforms.ToPILImage(), # Converte array numpy (H, W, C) para PIL Image
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(), # Converte PIL (H, W, C) para Tensor (C, H, W) e normaliza [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PhotosensitivityDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=30):

        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Mapeia os nomes das pastas para os rótulos inteiros
        # "FAIL" (perigoso) = 1 (classe positiva)
        # "PASS" (seguro)   = 0 (classe negativa)
        self.class_to_label = {"FAIL": 1, "PASS": 0}
        
        # armazena tuplas (caminho_do_arquivo, label_int)
        self.samples = [] 

        print(f"Buscando arquivos em {root_dir}...")
        
        # Itera sobre os nomes das classes (pastas) que esperamos
        for class_name, label in self.class_to_label.items():
            class_folder_path = os.path.join(root_dir, class_name)
            
            if not os.path.isdir(class_folder_path):
                print(f"Aviso: Pasta '{class_folder_path}' não encontrada.")
                continue
            
            # Lista todos os arquivos .npy dentro da pasta da classe
            for npy_file_name in os.listdir(class_folder_path):
                if npy_file_name.endswith(".npy"):
                    npy_path = os.path.join(class_folder_path, npy_file_name)
                    # Adiciona o caminho completo e o rótulo inteiro à lista
                    self.samples.append((npy_path, label))
                    
        if not self.samples:
            raise RuntimeError(f"Nenhum arquivo .npy encontrado em {root_dir}. "
                               f"Verifique a estrutura de pastas (ex: {root_dir}/PASS/clip1.npy)")

        print(f"Encontrados {len(self.samples)} arquivos.")


    def __len__(self):
        # Retorna o número total de amostras (arquivos .npy) encontradas
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Obter caminho do arquivo e rótulo da nossa lista 'self.samples'
        npy_path, label = self.samples[idx]
        
        # 2. Carregar o array .npy
        # Shape esperado: (F, H, W, C)
        try:
            video_data = np.load(npy_path)
        except Exception as e:
            print(f"Erro ao carregar {npy_path}: {e}")
            video_data = np.zeros((self.sequence_length, 224, 224, 3), dtype=np.uint8)
            label = 0 # Assume como "seguro" para não propagar erro, talvez seja mais interessante assumir como perigoso dependendo do caso

        # 3. Ajustar a sequência de frames (Subamostragem ou Padding)
        frames = self._process_frames(video_data)
        
        # 4. Aplicar transformações
        if self.transform:
            frames_transformed = [self.transform(frame) for frame in frames]
        else:
            frames_transformed = [torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) for frame in frames]

        # 5. Empilhar os frames transformados em um único tensor
        # Forma final: (sequence_length, C, H, W)
        input_tensor = torch.stack(frames_transformed)
        
        return input_tensor, torch.tensor(label, dtype=torch.float)

    def _process_frames(self, video_data):
        # Amostra ou preenche os frames para atingir a sequence_length.
        total_frames = video_data.shape[0]
        
        if total_frames == self.sequence_length:
            return video_data
        
        elif total_frames > self.sequence_length:
            start_idx = np.random.randint(0, total_frames - self.sequence_length + 1)
            return video_data[start_idx : start_idx + self.sequence_length]
        
        else:
            padding_needed = self.sequence_length - total_frames
            last_frame = video_data[-1:] 
            padding = np.repeat(last_frame, padding_needed, axis=0)
            return np.concatenate((video_data, padding), axis=0)

# ============================
# 2. Modelo (Com Dropout Ajustado)
# ============================
import torch
import torch.nn as nn
import torchvision.models as models

class Rede(nn.Module):
    # --- Dropout padrão aumentado para 0.6 para forçar generalização ---
    def __init__(self, cnn_output_size=512, lstm_hidden_size=256, lstm_num_layers=2, classifier_dropout=0.6):

        # Inicializa o modelo.
        # O classifier_dropout padrão foi aumentado para 0.6 para maior regularização.
    
        super(Rede, self).__init__()

        # --- 1. O Encoder CNN ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Permite o fine-tuning das camadas finais
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        num_features = resnet.fc.in_features 
        resnet.fc = nn.Identity() 
        self.cnn_extractor = resnet
        
        # --- 2. Processador Sequencial LSTM ---
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # --- 3. Classifier Head ---
        classifier_input_size = lstm_hidden_size * 2 
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout), 
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        # x (Tensor): (batch_size, sequence_length, C, H, W)
        batch_size, sequence_length, C, H, W = x.shape
        
        x = x.view(batch_size * sequence_length, C, H, W)
        features = self.cnn_extractor(x)
        features = features.view(batch_size, sequence_length, -1)
        
        lstm_out, _ = self.lstm(features)
        
        # Max Pooling sobre o tempo
        agg_features = lstm_out.permute(0, 2, 1) 
        agg_features = nn.functional.max_pool1d(agg_features, kernel_size=sequence_length)
        agg_features = agg_features.squeeze(dim=2)
        
        output = self.classifier(agg_features)
        
        return output

# ============================
# 3. Função para calcular acurácia e loss 
# ============================
def avaliar_modelo(model, dataloader, criterion, device):
    model.eval() # Modo de avaliação (desliga dropout, batchnorm etc.)
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # Desativa o cálculo de gradientes para economizar memória e tempo
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1) # GARANTE O SHAPE (B, 1)

            outputs = model(X_batch) # Saída são logits
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # Para calcular acurácia com BCEWithLogitsLoss:
            # Aplica sigmoid para obter probabilidades e depois compara com 0.5
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).long() # Converte probabilidades para 0 ou 1

            correct += (predicted == y_batch.long()).sum().item() # y_batch.long() para comparação
            total += y_batch.size(0)

    
    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc

# ============================
# 4. Loop de treino 
# ============================

def train_looping(model, train_loader, val_loader, criterion, writer, device):
    
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())

    # Adicionado weight_decay (regularização L2)
    optimizer = optim.Adam(params_to_update, lr=0.001, weight_decay=1e-4)
    
    epochs = 10 # Versão atual que estamos usando o melhor modelo foi da época 7 e a partir da época 8 começou a overfittar

    model.to(device) 

    # --- Lógica para salvar o melhor modelo ---
    best_val_loss = np.inf 
    os.makedirs("models", exist_ok=True)
    
    best_model_path = "models/model_v5.pth" #agora regularizado

    print(f"Iniciando novo treinamento agora com regularização...")
    print(f"O melhor modelo será salvo em: {best_model_path}")

    for epoch in range(epochs):
        model.train() 
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        # (O loop de treino interno permanece o mesmo)
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1) 
            
            optimizer.zero_grad()
            outputs = model(X_batch) 
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).long()
            correct_train += (predicted == y_batch.long()).sum().item()
            total_train += y_batch.size(0)

        train_loss_avg = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        val_loss, val_acc = avaliar_modelo(model, val_loader, criterion, device)
        
        print(f"Época {epoch+1:02d}, "
              f"Loss Treino: {train_loss_avg:.4f}, Acurácia Treino: {train_acc*100:.2f}%, "
              f"Loss Val: {val_loss:.4f}, Acurácia Val: {val_acc*100:.2f}%")

        writer.add_scalars("Losses", {"Train": train_loss_avg, "Validation": val_loss}, epoch)
        writer.add_scalars("Accuracies", {"Train": train_acc, "Validation": val_acc}, epoch)

        # --- Lógica para salvar o checkpoint do melhor modelo ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            print(f"  ✨ Nova melhor pontuação! Loss de Validação: {best_val_loss:.4f}. Salvando modelo...")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)

    print(f"\nTreinamento concluído.")
    print(f"💾 O melhor modelo V4 foi salvo em: {best_model_path} (com loss de {best_val_loss:.4f})")

    # Carrega os pesos do melhor modelo salvo antes de retornar
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo retornado agora é o da melhor época ({checkpoint['epoch']+1}).")
    except Exception as e:
        print(f"Aviso: Não foi possível recarregar o melhor modelo no final: {e}")
        
    return model

# ============================
# 5. Main
# ============================
def main():
    # Definir o dispositivo (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset_root_dir = "datasets/dataset_pse_npy_v2" 
    SEQ_LENGTH = 30 
    
    full_dataset = PhotosensitivityDataset(
        root_dir=dataset_root_dir, 
        transform=transform, 
        sequence_length=SEQ_LENGTH
    )

    total_len = len(full_dataset) 
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len 

    lengths = [train_len, val_len, test_len]
    print(f"Dataset total: {total_len} clipes .npy.")
    print(f"Divisão: Treino {lengths[0]}, Validação {lengths[1]}, Teste {lengths[2]}")

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, lengths)

    BATCH_SIZE = 16 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # --- FIM DA LÓGICA DE DADOS ---

    writer = SummaryWriter(log_dir="runs/treino3") 
    
    model = Rede()
    
    criterion = nn.BCEWithLogitsLoss() 

    trained_model = train_looping(model, train_loader, val_loader, criterion, writer, device)

    print(f"\n=== Avaliação final no conjunto de teste ===")
    test_loss, test_acc = avaliar_modelo(trained_model, test_loader, criterion, device)
    print(f"Loss teste: {test_loss:.4f}")
    print(f"Acurácia teste: {test_acc*100:.2f}%")

    writer.close()

if __name__ == "__main__":
    main()