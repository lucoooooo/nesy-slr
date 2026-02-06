# EDWARD
## Aggiornamenti applicati

///

## Build & Run Docker

```bash
    docker build -t edward . 
    docker run -it -v ./workspace:/workspace -w /workspace edward
```

## Run Example

```console
usage: main.py [--epochs EPOCHS] [--batch_size_train BATCH_SIZE_TRAIN]
               [--lr LR] [--seed SEED] [--batch_size_test BATCH_SIZE_TEST]
               [--no_train] [--modeldir MODELDIR] [--datadir DATADIR]
               
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
```

Una volta avviato il container Docker, per riprodurre i risultati basta eseguire:

```bash
    python3 main.py --no_train
```

## Riferimento bibliografico

**D. Tran, M. D. Hoffman, D. Moore, C. Suter, S. Vasudevan, A. Radul, M. Johnson, and R. A. Saurous** (2018).
"Simple, Distributed, and Accelerated Probabilistic Programming".
Neural Information Processing Systems (NeurIPS).
[Link EDWARD2](https://arxiv.org/abs/1811.02091) / [Link repo](https://github.com/google/edward2/)
**D. Tran, M. D. Hoffman, R. A. Saurous, E. Brevdo, K. Murphy, and D. M. Blei** (2017).
"Deep probabilistic programming".
International Conference on Learning Representations (ICLR).
[Link EDWARD](https://arxiv.org/abs/1701.03757) / [Link repo](https://github.com/blei-lab/edward)

