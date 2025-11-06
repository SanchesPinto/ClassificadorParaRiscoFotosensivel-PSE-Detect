import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
import argparse
import os

# ===================================================================
# 1. DEFINIÇÕES DO MODELO 
# ===================================================================

# Define o tamanho da imagem e a sequência (DEVE SER IGUAL AO TREINO)
IMG_SIZE = 224 
SEQ_LENGTH = 30 

# Transformações 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Definição da classe da Rede
class Rede(nn.Module):
    def __init__(self, cnn_output_size=512, lstm_hidden_size=256, lstm_num_layers=2, classifier_dropout=0.5):
        super(Rede, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer3.parameters():
            param.requires_grad = True 
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        num_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.cnn_extractor = resnet
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        classifier_input_size = lstm_hidden_size * 2 
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.shape
        x = x.view(batch_size * sequence_length, C, H, W)
        features = self.cnn_extractor(x)
        features = features.view(batch_size, sequence_length, -1)
        lstm_out, _ = self.lstm(features)
        agg_features = lstm_out.permute(0, 2, 1) 
        agg_features = nn.functional.max_pool1d(agg_features, kernel_size=sequence_length)
        agg_features = agg_features.squeeze(dim=2)
        output = self.classifier(agg_features)
        return output

# ===================================================================
# 2. FUNÇÕES DE PROCESSAMENTO E INFERÊNCIA
# ===================================================================

def _process_frames_for_inference(frames_array, sequence_length):

    # Aplica a mesma lógica de padding/amostragem do Dataset para um único clipe de inferência.

    total_frames = frames_array.shape[0]
    
    if total_frames == sequence_length:
        return frames_array
    
    elif total_frames > sequence_length:
        # Para inferência, pegamos os PRIMEIROS 'sequence_length' frames
        # (Em vez de um clipe aleatório, para a saída ser consistente)
        return frames_array[:sequence_length]
    
    else:
        # Vídeo mais curto: preenche (pad) repetindo o último frame
        padding_needed = sequence_length - total_frames
        last_frame = frames_array[-1:] 
        padding = np.repeat(last_frame, padding_needed, axis=0)
        return np.concatenate((frames_array, padding), axis=0)

def load_video_and_prepare_tensor(video_path, transform, sequence_length, device):

    # Carrega um vídeo, extrai frames, aplica a lógica do dataset
    # Transforma de .mp4 para .npy e aplica transformações

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: {video_path}")
        
    all_frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Converte de BGR (OpenCV) para RGB (PIL/PyTorch)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
    finally:
        cap.release()

    if not all_frames:
        raise ValueError(f"Nenhum frame lido do vídeo: {video_path}")

    # Converte a lista de frames em um array numpy
    frames_np = np.array(all_frames)
    
    # Aplica a mesma lógica de padding/amostragem do treino
    processed_frames = _process_frames_for_inference(frames_np, sequence_length)
    
    # Aplica as transformações de treino em cada frame
    frames_transformed = [transform(frame) for frame in processed_frames]
    
    # Empilha os frames em um único tensor
    input_tensor = torch.stack(frames_transformed)
    
    # Adiciona a dimensão do "batch" (o modelo espera B, S, C, H, W)
    input_tensor = input_tensor.unsqueeze(0) # Shape: [1, S, C, H, W]
    
    return input_tensor.to(device)

def run_inference(model_path, video_path):

    # Carrega o modelo e executa a previsão

    # 1. Definir o dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 2. Carregar a arquitetura do modelo
    model = Rede().to(device)
    
    # 3. Carregar os pesos treinados (checkpoint)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Erro ao carregar o modelo de '{model_path}'.")
        print("Certifique-se de que o caminho está correto e o arquivo não está corrompido.")
        print(f"Detalhe do erro: {e}")
        return

    # 4. Colocar o modelo em modo de avaliação (desliga dropout, etc.)
    model.eval()

    # 5. Mapeamento de classe (o oposto do seu dataset)
    # Lembre-se: 1 = "FAIL" (Alto Risco), 0 = "PASS" (Baixo Risco)
    class_names = {0: "Baixo Risco", 1: "Alto Risco"}
    
    try:
        # 6. Preparar o tensor do vídeo
        print(f"Processando vídeo: {video_path}...")
        input_tensor = load_video_and_prepare_tensor(video_path, transform, SEQ_LENGTH, device)

        # 7. Executar a inferência 
        with torch.no_grad():
            logit = model(input_tensor) # Saída é um tensor logit, ex: tensor([[2.7]])
            
            # --- INÍCIO DA CORREÇÃO ---
            
            # 1. Converte logit em probabilidade (AINDA COMO TENSOR)
            probabilities_tensor = torch.sigmoid(logit) # ex: tensor([[0.93]])
            
            # 2. Converte a probabilidade (tensor) em classe 0 ou 1 (tensor)
            # (probabilities_tensor > 0.5) vira tensor([[True]])
            # .long() converte tensor([[True]]) para tensor([[1]])
            predicted_tensor = (probabilities_tensor > 0.5).long()
            
            # 3. Agora, extraímos os valores Python dos tensores
            probability = probabilities_tensor.item() # ex: 0.93
            predicted_class_idx = predicted_tensor.item() # ex: 1
            
            # --- FIM DA CORREÇÃO ---
            
            predicted_class_name = class_names[predicted_class_idx]
            
            # 8. Mostrar o resultado
            print("\n--- Resultado da Análise ---")
            print(f"  Vídeo: {os.path.basename(video_path)}")
            print(f"  Previsão: {predicted_class_name}")
            print(f"  Confiança (Probabilidade de ser 'Alto Risco'): {probability*100:.2f}%")

    except Exception as e:
        print(f"\n--- ERRO DURANTE A INFERÊNCIA ---")
        print(f"Não foi possível processar o vídeo: {video_path}")
        print(f"Detalhe do erro: {e}")

# ===================================================================
# 3. SCRIPT PRINCIPAL (Interface de Linha de Comando)
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de inferência para o modelo PSE-Detect."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Caminho para o arquivo do modelo treinado (.pth)."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Caminho para o arquivo de vídeo de entrada (ex: video.mp4)."
    )
    
    args = parser.parse_args()
    
    # Chama a função principal de inferência
    run_inference(args.model, args.input)