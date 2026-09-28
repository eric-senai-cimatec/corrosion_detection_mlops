# 🚀 corrosion_detection_mlops

A corrosion detection solution using AI and MLOps.

---

## 📹 Fase 1: Coleta em Tempo Real & Servimento Dinâmico

Esta fase inicializa a aplicação de produção. O servidor FastAPI localiza autonomamente o modelo `@champion` no banco SQLite. O cliente OpenCV captura o stream da webcam e usa um pool de threads em segundo plano para enviar frames, filtrando e salvando capturas em `online_data/` apenas se houver detecção acima de **0.30 de confiança**.

Para testar o servimento, abra **dois terminais separados** na raiz do projeto:

### Terminal 1
*Inicializa o servidor FastAPI e carrega o modelo `@champion` ativo:*
```bash
python src/serving/app.py
```

### Terminal 2
*Inicializa a captura da webcam com processamento assíncrono em background:*
```bash
python src/serving/client_webcam.py
```

---

## 🎨 Fase 2: Curadoria Humana & Active Learning (Human-in-the-Loop)

Com os novos frames reais de produção armazenados na pasta `online_data/`, o pipeline aciona a etapa de aceleração de anotação. O modelo gera pré-marcações no formato JSON aceito pelo **Labelme**, permitindo que o operador humano atue apenas como revisor e corretor dos boxes duvidosos.

```bash
# 1. Processa as novas imagens e gera os arquivos JSON de pré-anotação automática
python -m src.data_automation.send_to_review

# 2. Abre a interface gráfica do Labelme na pasta para auditoria e ajuste dos boxes
labelme review_data

# 3. Traduz os JSONs corrigidos para TXT (padrão YOLO) e consolida os dados na pasta de treino
python -m src.data_automation.parse_review_to_train
```

---

## 📦 Fase 3: Versionamento de Dados & Linhagem (DVC)

Após a consolidação dos novos dados na pasta oficial de treinamento (`data/`), é obrigatório registrar o novo estado imutável do dataset. O DVC calcula a nova assinatura digital (hash MD5) e sincroniza os dados físicos pesados diretamente com o Google Drive privado.

```bash
# 4. Atualiza o arquivo de ponteiro leve local com o novo hash do dataset revisado
dvc add data

# 5. Faz o upload seguro das novas imagens e labels brutos para o repositório remoto (Google Drive)
dvc push

# 6. Registra as alterações do ponteiro do DVC no histórico de controle de versão do código
git add data.dvc
git commit -m "chore: adiciona novos frames de corrosao reais ao dataset de treino"
```

---

## 🏆 Fase 4: Treinamento, Governança Estrita & Atualização Viva

Com os dados blindados na nuvem, inicia-se o ciclo de engenharia de Machine Learning. O script de avaliação executa um portão de qualidade multi-métrica. O modelo novo só assume o posto de `@champion` se passar pelos thresholds mínimos e superar o score combinado do antigo campeão. Se aprovado, a API em produção é atualizada em tempo real via HTTP.

```bash
# 7. Dispara o treinamento do YOLO no Windows gerenciando os subprocessos do DataLoader
python -m src.model_train.yolo

# 8. Executa a validação no split de teste e aplica as travas de negócio (mAP50 > 50% & Recall > 40%)
python -m src.model_eval.yolo

# 9. Atualização Viva: Envia um sinal HTTP para a API recarregar o novo modelo sem gerar downtime
curl -X POST http://127.0.0

# 10. Inicia o painel gráfico do MLflow para auditoria de desempenho e governança do ciclo de vida
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*💡 Dica: Após rodar o comando 10, acesse o endereço **`http://127.0.0.1:5000`** no seu navegador para auditar o status das versões na aba **Models**.*
