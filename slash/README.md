# SLASH
## Aggiornamenti applicati

./SLASH/src/SLASH/slash.py → aggiornato alcuni pezzi hardcoded per cuda

## Build & Run Docker

```bash
    docker build -t slash . 
    docker run -it -v ./workspace:/workspace -w /workspace slash
```

## Run Example

```console
usage: main.py [--epochs EPOCHS] [--batch_size_train BATCH_SIZE_TRAIN]
               [--lr LR] [--seed SEED] [--batch_size_test BATCH_SIZE_TEST]
               [--no_train] [--modeldir MODELDIR] [--datadir DATADIR]
               [--k K] [--method METHOD] [-p_num P_NUM]
               
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

  --k K                 Numero Top-K prove(default: 0)

  --method {"exact","top_k","same"}  
                        Metodo di inferenza 
                        (default: same)

  --p_num P_NUM         Numero di core da usare nel training 
                        (default: 8)
```

Una volta avviato il container Docker, per riprodurre i risultati basta eseguire:

```bash
    python3 main.py --no_train
```

## Riferimento bibliografico

**Skryagin, A., Ochs, D., Dhami, D.S., & Kersting, K.** (2023).  
   *Scalable Neural-Probabilistic Answer Set Programming* (SLASH).  
   Journal of Artificial Intelligence Research.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85178997369&doi=10.1613%2fJAIR.1.15027&partnerID=40&md5=7d949c308cfce51ed99607c371aaab7c) /
   [Link al repo](https://github.com/ml-research/SLASH)
