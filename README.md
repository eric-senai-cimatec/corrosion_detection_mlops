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
*💡 Dica: Aponte a webcam para focar em cenários de teste com corrosão. Quando o terminal indicar que os frames foram gerados e salvos em `online_data/`, pressione a tecla **'q'** na janela de vídeo para encerrar.*

---

## 🧭 Fase 2: Monitoramento & Orquestração Automatizada (Apache Airflow - Parte I)

Com os dados de produção capturados na pasta `online_data/`, o Apache Airflow assume o papel de maestro. O monitoramento calcula se houve desvio estatístico (*Data Drift*) nas propriedades de brilho, contraste e cor das novas imagens utilizando o **Evidently AI**.

### Passo 1: Inicializar o Ambiente do Airflow
Antes de rodar a esteira, garanta que o seu servidor local do Apache Airflow esteja de pé e com as DAGs cadastradas:
```bash
# Inicializa o webserver e o scheduler do Airflow
airflow db init
airflow users create --username admin --firstname Eric --lastname Santos --email admin@mlops.com --role Admin --password admin
airflow webserver --port 8080
airflow scheduler
```
*💡 Acesse a interface web em `http://localhost:8080` com as credenciais criadas.*

### Passo 2: Executar a DAG 1 (Monitoramento)
A primeira DAG (`corrosion_phase1_monitoring`) executa a análise de drift e decide os próximos passos:
1. Roda o script de observabilidade: `python -m src.observability.monitor`.
2. Se o desvio atingir o limite crítico, o `BranchPythonOperator` dispara o script de Active Learning (`python -m src.data_automation.send_to_review`), gerando as pré-anotações automáticas e deixando os arquivos em `review_data/`.
3. Se não houver drift, o pipeline encerra de forma limpa na própria interface do Airflow para poupar processamento.

---

## 🎨 Fase 3: Curadoria Humana & Active Learning (Human-in-the-Loop)

Caso o Airflow tenha detectado Drift na fase anterior, as novas imagens com defeito estarão aguardando a sua revisão humana na pasta `review_data/` acompanhadas de pré-anotações inteligentes em JSON geradas pelo modelo campeão.

```bash
# 1. Abre a interface gráfica do Labelme apontando para a fila de revisão humana
labelme review_data

# 2. Quando terminar de corrigir/aprovar os boxes e clicar em salvar, execute o conversor:
python -m src.data_automation.parse_review_to_train
```
*💥 MÁGICA DE MLOps:* O script `parse_review_to_train.py` traduzirá os JSONs para arquivos `.txt` padrão YOLO, mesclará tudo com a sua base oficial de treino, limpará as pastas e **disparará de forma 100% automatizada um Webhook (API REST HTTP) que acorda a segunda DAG do Airflow** para processar o retreino, sem necessidade de intervenção humana no painel do orquestrador.

---

## 🏆 Fase 4: Sincronização, Retreino & Atualização Viva (Apache Airflow - Parte II)

A segunda DAG (`corrosion_phase2_retrain`) acorda imediatamente após o recebimento do sinal HTTP disparado pelo seu script de curadoria. Ela gerencia de ponta a ponta as tarefas pesadas de computação, governança e implantação contínua:

### 4.1 Versionamento de Dados (DVC)
*O Airflow executa automaticamente o isolamento das novas assinaturas imutáveis dos dados na nuvem:*
```bash
# Atualiza o hash MD5 da pasta 'data' localmente no arquivo data.dvc e envia para o Google Drive
dvc add data
dvc push

# Registra as alterações do ponteiro do DVC no controle de versão do histórico do Git
git add data.dvc
git commit -m "chore: adiciona novos frames de corrosao reais ao dataset de treino via pipeline"
```

### 4.2 Treinamento, Governança Estrita & Deploy Contínuo (MLflow)
*O Airflow dispara o ciclo de modelagem, avalia os resultados em relação ao campeão atual e atualiza a API em produção de forma viva:*
```bash
# 1. Inicia o treinamento do YOLO no Windows lendo as configurações do model_config.yaml
python -m src.model_train.yolo

# 2. Executa a validação no split de teste filtrando as pastas pelo relógio do sistema (getmtime)
# Aplica as travas de negócio multi-métrica (mAP50 > 50% & Recall > 40%) contra o Champion atual
python -m src.model_eval.yolo

# 3. CD (Continuous Deployment): Envia um sinal HTTP de recarregamento para o servidor de produção
# A sua API FastAPI atualiza os pesos da memória em tempo real SEM DERRUBAR o sistema!
curl -X POST http://127.0.0

# 4. Inicia o painel gráfico do MLflow para auditoria e histórico de execuções
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*💡 Nota: Acesse `http://127.0.0.1:5000` no seu navegador para verificar as tabelas comparativas das Runs e confirmar graficamente que a nova versão assumiu a etiqueta de **`champion`**.*
