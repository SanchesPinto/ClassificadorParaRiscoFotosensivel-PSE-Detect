# ClassificadorParaRiscoFotosens-vel-PSE-Detect
IA para classificar risco de epilepsia fotossensível (EFS) em vídeos, seguindo a norma ITU-R BT.1702-3. O modelo (CNN+LSTM) foi treinado em um dataset sintético de arrays .npy, gerado e validado pela ferramenta IRIS (EA) para garantir a segurança e precisão dos rótulos. Projeto focado em acessibilidade de mídia digital.
## 🚀 Como Executar (Ambiente Local)

Siga estas etapas para configurar e treinar o modelo em sua máquina local.

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/SanchesPinto/ClassificadorParaRiscoFotosens-vel-PSE-Detect.git
    cd ClassificadorParaRiscoFotosensivel-PSE-Detect
    ```

2.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # No Linux/macOS
    # ou
    .\venv\Scripts\activate   # No Windows
    ```

3.  **Instale as Dependências:**
    
    ```bash
    pip install torch torchvision numpy pandas tensorboard
    # ou
    pip install -r requirements.txt
    ```

4.  **Estrutura dos Dados:**
    Certifique-se de que seu dataset (`.npy`) esteja seguindo a estrutura de pastas esperada:
    ```
    /seu-projeto/
    ├── datasets/
    ├   ├── dataset_pse_npy/
    │       ├── FAIL/
    │       │   ├── fail_flash_001.npy
    │       │   └── ...
    │       └── PASS/
    │           ├── pass_seguro_001.npy
    │           └── ...
    ├── train.py
    └── ...
    ```

5.  **Ajuste o Script (se necessário):**
    Abra o arquivo `train.py` e verifique se as seguintes variáveis dentro da função `main()` estão corretas para o seu ambiente:
    * `dataset_root_dir`: Deve apontar para o nome da sua pasta de dataset (ex: `"dataset_pse_npy"`).
    * `log_dir`: Onde os logs do TensorBoard serão salvos (ex: `"runs/experiment_local"`).
    * `model_save_dir` (na função `train_looping`): Onde os modelos (`.pth`) serão salvos (ex: `"models"`).

6.  **Execute o Treinamento:**
    ```bash
    python3 train.py
    ```
    O script irá detectar automaticamente seu dispositivo (CPU ou GPU, se disponível e configurada) e iniciar o treinamento.

7.  **Monitore com o TensorBoard:**
    Enquanto o `train.py` está rodando, abra um **novo terminal** no mesmo diretório e execute:
    ```bash
    tensorboard --logdir=runs
    ```
    Abra o link local (geralmente `http://localhost:6006/`) no seu navegador para ver as curvas de *Loss* e Acurácia em tempo real.
