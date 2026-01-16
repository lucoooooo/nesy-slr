# DeepProbLog
## Aggiornamenti applicati

./deepproblog/src/deepproblog/train.py → aggiunto return di loss per epoca

## Build & Run Docker

```bash
    docker build -t deepproblog . 
    docker run -it -v ./workspace:/workspace -w /workspace deepproblog
```

## Run Example

```console
usage: main.py [--epochs EPOCHS] [--batch_size_train BATCH_SIZE_TRAIN]
               [--lr LR] [--seed SEED] [--batch_size_test BATCH_SIZE_TEST]
               [--no_train] [--modeldir MODELDIR] [--datadir DATADIR]
               [--method {exact,approximate}]

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

  --method {exact,approximate}
                        Metodo d'inferenza. (default: approximate)
```
Una volta avviato il container Docker, per riprodurre i risultati basta eseguire:

```bash
    python3 main.py --no_train
```

## Riferimento bibliografico
**Manhaeve, R., Dumančić, S., Kimmig, A., Demeester, T., & De Raedt, L.** (2021).  
    *Neural probabilistic logic programming in DeepProbLog*.  
    Artificial Intelligence.  
    [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85104453726&doi=10.1016%2fj.artint.2021.103504&partnerID=40&md5=9b46926a1251f8176c535e6027367e8f) /
    [Link al repo](https://github.com/ML-KULeuven/deepproblog)