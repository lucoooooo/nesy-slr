# DeepStochLog
## Aggiornamenti applicati

./deepstochlog/deepstochlog/trainer.py → aggiunto return di loss e accuracy per epoca
./deepstochlog/deepstochlog/dataloader.py → inizializzata variabile self.length

## Build & Run Docker

```bash
    docker build -t deepstochlog . 
    docker run -it -v ./workspace:/workspace -w /workspace deepstochlog
```

## Run Example

```console
usage: main.py [--epochs EPOCHS] [--batch_size_train BATCH_SIZE_TRAIN]
               [--lr LR] [--seed SEED] [--batch_size_test BATCH_SIZE_TEST]
               [--no_train] [--modeldir MODELDIR] [--datadir DATADIR]
               [--max-size MAX_SIZE] 
               
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

  --max-size MAX_SIZE
                        Massima dimensione del dataset neuro-simbolico.
                        (default: 15000 --> dataset MNIST completo)
```
Una volta avviato il container Docker, per riprodurre i risultati basta eseguire:

```bash
    python3 main.py --no_train
```

## Riferimento bibliografico
**Winters, T., Marra, G., Manhaeve, R., & De Raedt, L.** (2022).  
   *DeepStochLog: Neural Stochastic Logic Programming*.  
   Proceedings of the 36th AAAI Conference on Artificial Intelligence, AAAI 2022.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85137088750&doi=10.1609%2faaai.v36i9.21248&partnerID=40&md5=602ba9420d8c16993c33662bd9656d10) /
   [Link al repo](https://github.com/ML-KULeuven/deepstochlog)