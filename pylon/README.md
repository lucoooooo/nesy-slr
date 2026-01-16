# PYLON
## Aggiornamenti applicati

./pylon/pylon/utils.py → aggiunto "value = value.to(log_prob.device)" per non avere mismatch di device cuda / cpu
./pylon/pylon/brute_force_solve.py → aggiunto **kwargs a riga 50 (fix preso da pull request nel repository pylon)

## Build & Run Docker

```bash
    docker build -t pylon . 
    docker run -it -v ./workspace:/workspace -w /workspace pylon
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
**Ahmed, K., Li, T., Ton, T., Guo, Q., Chang, K.-W., Kordjamshidi, P., Srikumar, V., Van den Broeck, G., & Singh, S.** (2022).  
   *PYLON: A PyTorch Framework for Learning with Constraints*.  
   Proceedings of the 36th AAAI Conference on Artificial Intelligence, AAAI 2022.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85127722724&doi=10.1609%2faaai.v36i11.21711&partnerID=40&md5=2a73bf8b9df4878502ad57556bd3710a) /
   [Link al repo](https://github.com/pylon-lib/pylon)