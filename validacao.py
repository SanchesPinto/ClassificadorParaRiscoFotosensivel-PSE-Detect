import cv2
import numpy as np
import os
import glob
import subprocess 
import json      
import shutil    


# Caminho para o IRIS compilado
IRIS_EXECUTABLE_PATH = r""

# --- 2. CONFIGURAÇÃO DOS DADOS ---
NPY_DATA_DIRS = ["datasets/dataset_pse_npy_v3/FAIL", "datasets/dataset_pse_npy_v3/PASS"]  # Pastas contendo os arquivos .npy para validação
TEMP_DIR = "temp_pipeline_work"
IRIS_OUTPUT_DIR = "Results" # O IRIS cria esta pasta por padrão

# Configurações de vídeo 
FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

def convert_npy_to_video(npy_path, output_video_path):
    # Converte um .npy em um vídeo .mp4 temporário
    try:
        video_data = np.load(npy_path)
        height, width, _ = video_data[0].shape
        
        writer = cv2.VideoWriter(output_video_path, FOURCC, FPS, (width, height))
        for frame in video_data:
            writer.write(frame)
        writer.release()
    except Exception as e:
        print(f"  [ERRO] Falha ao converter {npy_path}: {e}")
        return False
    return True

def parse_iris_output(json_path):

    # Retorna 'FAIL' (perigoso) se qualquer falha for detectada
    # Retorna 'PASS' (seguro) se nenhuma falha for encontrada

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Tentativa 1: O próprio arquivo é uma lista de frames
        frame_list = data
        if isinstance(data, dict) and 'frames' in data:
             # Tentativa 2: É um dicionário com uma chave 'frames'
            frame_list = data['frames']
            
        if not isinstance(frame_list, list):
             print(f"  [AVISO] Formato JSON inesperado em {json_path}. Não foi possível parsear.")
             return "PARSE_ERROR"

        for frame in frame_list:
            # De acordo com o README: 0 = pass, 1 = warning, 2 = extended fail, 3 = flash fail
            if (frame.get("Luminance Frame Result", 0) > 0 or
                frame.get("Red Frame Result", 0) > 0 or
                frame.get("Pattern Frame Result", 0) > 0):
                return "FAIL (Perigoso)" # Detectou um problema

        return "PASS (Seguro)" # Nenhum frame falhou
    
    except json.JSONDecodeError:
        print(f"  [ERRO] O arquivo JSON de resultado está corrompido ou vazio: {json_path}")
        return "JSON_ERROR"
    except Exception as e:
        print(f"  [ERRO] Erro ao ler o JSON {json_path}: {e}")
        return "PARSE_ERROR"

def main():
    # Pipeline de validação automatizada
    
    # Verifica se o executável do IRIS existe
    if not os.path.exists(IRIS_EXECUTABLE_PATH):
        print(f"!!! ERRO CRÍTICO !!!")
        print(f"O executável do IRIS não foi encontrado em:")
        print(f"{IRIS_EXECUTABLE_PATH}")
        print("Por favor, atualize a variável 'IRIS_EXECUTABLE_PATH' no script.")
        return

    # Garante que a pasta de trabalho temporária exista e esteja limpa
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    # O IRIS gera uma pasta 'Results' no CWD (Current Working Directory)
    if os.path.exists(IRIS_OUTPUT_DIR):
        shutil.rmtree(IRIS_OUTPUT_DIR)

    validation_summary = []
    print("--- INICIANDO PIPELINE DE VALIDAÇÃO AUTOMATIZADA ---")

    for data_dir in NPY_DATA_DIRS:
        print(f"\nProcessando diretório: {data_dir}")
        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        
        if not npy_files:
            print("  Nenhum arquivo .npy encontrado.")
            continue

        for npy_path in npy_files:
            base_name = os.path.basename(npy_path).replace('.npy', '')
            print(f"  > Processando: {base_name}")
            
            temp_video_path = os.path.join(TEMP_DIR, f"{base_name}.mp4")
            
            # O IRIS salva o resultado em 'Results/nome_do_video.json'
            expected_json_path = os.path.join(IRIS_OUTPUT_DIR, f"{base_name}.json")
            
            status = "INICIADO"
            iris_result = "N/A"

            try:
                # 1. CONVERTER .npy para .mp4
                if not convert_npy_to_video(npy_path, temp_video_path):
                    status = "FALHA_CONVERSAO"
                    continue

                # 2. EXECUTAR ANÁLISE IRIS
                # Comando: ./iris_app -v video.mp4 -j true -p true
                iris_command = [
                    IRIS_EXECUTABLE_PATH,
                    "-v", temp_video_path,
                    "-j", "true",
                    "-p", "true" # Habilita detecção de padrão
                ]
                
                print(f"    Executando IRIS...")
                subprocess.run(iris_command, check=True, capture_output=True, text=True)
                print(f"    Análise IRIS concluída.")

                # 3. PARSEAR RESULTADO JSON
                if os.path.exists(expected_json_path):
                    iris_result = parse_iris_output(expected_json_path)
                    status = "CONCLUIDO"
                else:
                    print(f"  [ERRO] IRIS não gerou o arquivo de resultado esperado em: {expected_json_path}")
                    status = "FALHA_IRIS"

            except subprocess.CalledProcessError as e:
                print(f"  [ERRO] O executável do IRIS falhou:")
                print(e.stderr)
                status = "FALHA_EXECUCAO_IRIS"
            except Exception as e:
                print(f"  [ERRO] Um erro inesperado ocorreu: {e}")
                status = "FALHA_DESCONHECIDA"
            
            finally:
                # 4. LIMPEZA SEGURA (SEMPRE EXECUTA)
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                if os.path.exists(expected_json_path):
                    os.remove(expected_json_path)
                
                print(f"    Limpeza de arquivos temporários concluída.")

                validation_summary.append({
                    "arquivo": base_name,
                    "pasta_origem": data_dir,
                    "status_pipeline": status,
                    "resultado_iris": iris_result
                })

    # 5. LIMPEZA FINAL das pastas de trabalho
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    if os.path.exists(IRIS_OUTPUT_DIR):
        shutil.rmtree(IRIS_OUTPUT_DIR)
    print("\nLimpeza final das pastas de trabalho concluída.")

    # 6. EXIBIR RELATÓRIO FINAL
    print("\n\n--- RELATÓRIO FINAL DE VALIDAÇÃO ---")
    print(f"{'Arquivo':<40} | {'Pasta de Origem':<20} | {'Resultado IRIS':<15}")
    print("-" * 78)
    
    for item in validation_summary:
        print(f"{item['arquivo']:<40} | {item['pasta_origem']:<20} | {item['resultado_iris']:<15}")

if __name__ == "__main__":
    main()