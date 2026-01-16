# NeurASP
## Aggiornamenti applicati

./NeurASP/neurasp.py → aggiunto return di loss

## Build & Run Docker

```bash
    docker build -t neurasp . 
    docker run -it -v ./workspace:/workspace -w /workspace neurasp
```

## Run Example

```console
usage: main.py [--epochs EPOCHS] [--batch_size_train BATCH_SIZE_TRAIN]
               [--lr LR] [--seed SEED] [--batch_size_test BATCH_SIZE_TEST]
               [--no_train] [--modeldir MODELDIR] [--datadir DATADIR]
               [--max-size MAX_SIZE] [--method {exact, sampling}]
               
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
                        (default: 5000)

  --method {exact, sampling}
                        Metodo d'inferenza. (default: exact)
```
Una volta avviato il container Docker, per riprodurre i risultati basta eseguire:

```bash
    python3 main.py --no_train
```

## Riferimento bibliografico
**Yang, Z., Ishay, A., & Lee, J.** (2020).  
   *NeurASP: Embracing neural networks into answer set programming*.  
   IJCAI International Joint Conference on Artificial Intelligence.  
   [Link al paper](https://arxiv.org/pdf/2307.07700) /
   [Link al repo](https://github.com/azreasoners/NeurASP)