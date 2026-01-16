# SCALLOP
## Aggiornamenti applicati

///

## Build & Run Docker

```bash
    docker build -t scallop . 
    docker run -it -v ./workspace:/workspace -w /workspace scallop
```

## Run Example

```console
usage: main.py [--epochs EPOCHS] [--batch_size_train BATCH_SIZE_TRAIN]
               [--lr LR] [--seed SEED] [--batch_size_test BATCH_SIZE_TEST]
               [--no_train] [--modeldir MODELDIR] [--datadir DATADIR]
               [--k K] [--provenance METHOD]
               
  --epochs EPOCHS       Numero di epoche di training. (default: 20)
  --batch_size_train BATCH_SIZE_TRAIN
                        Input batch size per training. (default: 32)
  --lr LR               Learning rate per l'optimizer. (default: 0.001)
  --seed SEED           Seed randomico per la riproducibilità. (default: 123)
  --batch_size_test BATCH_SIZE_TEST
                        Input batch size per il testing/evaluation. (default: 32)

  --no_train            Se settato, salta la fase di training e esegue solo
                        la fase di testing. 
                        (default: False)

  --modeldir MODELDIR   Percorso per la directory dove vengono salvati i
                        modelli e da dove vengono caricati. 
                        (default: model/mnist_sum_2)

  --datadir DATADIR     Percorso per la directory contenente il dataset e i
                        risultati. 
                        (default: ./data)
  --k K                 Numero Top-K prove(default: 3)
  --provenance METHOD   Metodo di inferenza (default: "difftopkproofs")
```

Una volta avviato il container Docker, per riprodurre i risultati basta eseguire:

```bash
    python3 main.py --no_train
```

## Riferimento bibliografico
**Li, Z., Huang, J., & Naik, M.** (2023).  
   *Scallop: A Language for Neurosymbolic Programming*.  
   Proceedings of the ACM on Programming Languages.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85161998644&doi=10.1145%2f3591280&partnerID=40&md5=4c7ff83ac09ee7dbf4ce0a598e59c29a) /
   [Link al repo](https://github.com/scallop-lang/scallop)