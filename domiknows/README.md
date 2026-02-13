# DomiKnowS
## Aggiornamenti applicati

-> pulizia esempio originale mantenendo solo metodi PrimalDual e Sampling
-> se si volesse modificare il codice per renderlo compatibile con un solver non di Pytorch (quindi Gurobi), bisogna acquisire la licenza ([Leggi qui](https://github.com/HLR/DomiKnowS/blob/develop/GurobiREADME.md))

## Build & Run Docker

```bash
    docker build -t domiknows . 
    docker run -it -v ./workspace:/workspace -w /workspace domiknows
```

## Run Example
Questo script bash automatizza l'esecuzione e il confronto diretto tra un modello puramente neurale (Baseline) e l'approccio neuro-simbolico implementato con DomiKnowS (utilizzando il metodo di integrazione `PrimalDual`). 

Lo script si occupa di lanciare entrambi i modelli, effettuare il training (se richiesto), e infine avviare lo script di analisi per generare e salvare i risultati comparati (in ./results)
```console
./run_comparison.sh <LR> <DATA_PATH> <MODEL_SAVE_PATH> <BATCH_SIZE> <SEED> [NO_TRAIN]
```
Una volta avviato il container Docker, per riprodurre i risultati basta eseguire:

```bash
    ./run_comparison 1e-3 data model/mnist_sum_2 32 123 no_train
```
## Riferimento bibliografico

**Rajaby Faghihi, H., Guo, Q., Uszok, A., Nafar, A., Raisi, E., & Kordjamshidi, P.**  (2021). 
   *DomiKnowS: A Library for Integration of Symbolic Domain Knowledge in Deep Learning*. 
   In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. 
   Association for Computational Linguistics. [Link al paper](https://aclanthology.org/2021.emnlp-demo.27) / [Link al repo](https://github.com/HLR/DomiKnowS)


