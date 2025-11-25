# 🛡️ PSE-Detect: Detecção de Risco de Epilepsia Fotossensível com Deep Learning

> **Uma solução de IA para classificar automaticamente segmentos de vídeo que violam as normas de segurança ITU-R BT.1702-3 para epilepsia fotossensível.**

---

## 📋 Sobre o Projeto

A epilepsia fotossensível (EFS) afeta uma parcela significativa da população, sendo desencadeada por estímulos visuais específicos como flashes rápidos e padrões geométricos de alto contraste. A verificação manual de conteúdo de vídeo é inviável em escala.

Este projeto implementa uma arquitetura híbrida (**CNN + LSTM**) capaz de analisar características espaciais e temporais para classificar vídeos como **"Alto Risco"** ou **"Baixo Risco"** com alta precisão, servindo como uma ferramenta automatizada de acessibilidade e QA (Quality Assurance) para mídia digital.

### 🎯 Principais Diferenciais (Destaques de Engenharia)
* **Segurança e Ética de Dados:** Desenvolvimento de metodologia para treinar o modelo sem a necessidade de coletar/distribuir vídeos perigosos reais.
* **Domain Adaptation:** Resolução do problema de *Domain Gap* utilizando técnicas de *Data Augmentation* complexas (sobreposição em fundos reais).
* **Validação Rigorosa:** Uso de uma ferramenta industrial ("Oracle") para garantir a integridade dos rótulos sintéticos.

---

## ⚙️ Arquitetura e Pipeline de MLOps

O projeto foi estruturado seguindo princípios de reprodutibilidade e experimentação iterativa.

### 1. O Pipeline de Dados (Data Engineering)
Devido ao risco inerente aos dados, adotou-se uma abordagem de **Dados Sintéticos Validados**:

1.  **Geração:** Scripts Python geram arrays NumPy (`.npy`) contendo padrões estroboscópicos e geométricos parametrizados.
2.  **Validação (Ground Truth):** Cada amostra gerada é validada contra a ferramenta **IRIS (Electronic Arts)**, referência na indústria, para garantir que o rótulo (PASS/FAIL) respeita estritamente a norma ITU.
3.  **Segurança:** O dataset é mantido em formato não-executável (`.npy`) para prevenir visualização acidental.

### 2. O Modelo (Híbrido Espaço-Temporal)
* **Encoder Espacial (CNN):** ResNet-18 (pré-treinada na ImageNet) com *Fine-Tuning* nas camadas intermediárias (`layer2` a `layer4`) para extração de features visuais complexas.
* **Encoder Temporal (RNN):** LSTM Bidirecional para capturar a frequência e persistência dos flashes ao longo do tempo.
* **Agregador:** Camada de *Max-Pooling-Over-Time* que garante que um evento de risco detectado em *qualquer* momento do vídeo dispare o alerta de classificação global.

### 3. Estratégia de Treinamento e Monitoramento
* **Experiment Tracking:** Utilização do **TensorBoard** para monitoramento em tempo real de métricas de perda (Loss) e acurácia.
* **Regularização:** Aplicação de *Dropout* (0.6) e *Weight Decay* (L2) para combater overfitting.
* **Model Checkpointing:** Implementação de callbacks para *Early Stopping*, salvando o modelo no ponto de generalização máxima (mínimo *loss* de validação) antes da divergência.

---

## 📈 Resultados e Análise

O desenvolvimento passou por múltiplas iterações para superar o *overfitting* em dados sintéticos.

### O Desafio da Generalização
Inicialmente, o modelo atingiu 99.5% de acurácia em dados sintéticos (fundo preto), mas falhou em vídeos reais. Diagnosticou-se um problema de **Domain Gap**.
* **Solução:** Implementação de um gerador de dataset V3 que utiliza **188 vídeos de fundo reais** (paisagens, gameplays, vlogs) e realiza a sobreposição (*blending*) dos efeitos de risco, forçando o modelo a distinguir "sinal" de "ruído".

### Performance Final (Modelo V5)
O gráfico abaixo ilustra o treinamento final. Nota-se o ponto exato de *Early Stopping* (Época 7/8) onde o modelo atinge a melhor capacidade de generalização antes de iniciar o overfitting.

| Curvas de Loss (Treino vs Validação) | Curvas de Acurácia |
|:---:|:---:|
| ![Loss Graph](assets/image_f9aedb.png) | ![Accuracy Graph](assets/image_f9ae9c.png) |

### Teste em Cenário Real (Inferência)
O modelo final foi submetido a testes de estresse com vídeos notórios e clipes seguros.

![Resultados da Inferência](assets/image_f95845.png)

* ✅ **Porygon.mp4 (Caso Pokémon):** Detectado como **Alto Risco (96.74%)**.
* ✅ **Show de Luzes:** Detectado como **Alto Risco (95.70%)**.
* ✅ **Vídeos de Paisagem/Vlog:** Corretamente ignorados (**< 4%** de falso positivo).

---

## ⚠️ Nota sobre o Dataset

Por razões de segurança e conformidade com as diretrizes da plataforma, o dataset de treinamento contendo estímulos estroboscópicos e o código gerador não estão incluídos neste repositório público. O foco deste repositório é demonstrar a arquitetura do modelo, o pipeline de treinamento e a capacidade de inferência.