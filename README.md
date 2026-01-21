# Librerie e strumenti e applicazioni per l'IA neuro-simbolica: una panoramica 

Questo repository contiene il codice, gli esperimenti e la documentazione relativa alla mia Tesi di Laurea Triennale in Informatica, focalizzata sulla rassegna e l'analisi delle moderne librerie di **Intelligenza Artificiale Neuro-Simbolica (NeSy)**.

L'obiettivo del progetto è eseguire un confronto qualitativo seguendo una tassonomia riguardante gli aspetti linguistici, i meccanismi di inferenza, l'integrazione architetturale e dell'usabilità dei principali framework NeSy. Il codice qui presente contiene gli esperimenti che costituiscono il confronto quantitativo eseguito sul task **MNIST single digit addition**.

**_Perchè questo task?_** Analizzando la letteratura, emerge che tale problema è considerato un benchmark standard per la valutazione dei modelli in ambito neuro-simbolico.

## Librerie Analizzate

Il progetto prende in esame le seguenti librerie:

* **DeepProbLog** 
* **DeepStochLog** 
* **LTNtorch** 
* **NeurASP** 
* **PYLON** 
* **Scallop** 
* **SLASH** 

## Struttura del Repository

Il repository è organizzato in cartelle, ognuna contenente l'implementazione del task _MNIST single digit addition_ per la specifica libreria. Ogni esperimento è stato costruito in un ambiente isolato tramite Docker; di conseguenza, non è presente uno script bash per l'esecuzione centralizzata di tutti i task.

Nel README.md di ogni sottocartella è presente una gruida all'avvio del container Docker e all'esecuzione del codice. In alcuni casi è stata inclusa direttamente la versione clonata della libreria, poichè sono stati necessari lievi adattamenti ai file sorgente per garantire la corretta formattazione dei risultati.

```text
.
├── deepproblog/               
│   ├── deepproblog/ # libreria clonata dal repository ufficiale
│   ├── workspace/
│   │   ├── data # directory contenente il dataset e i risultati 
│   │   ├── model # directory contenente i modelli pretrainati
│   │   ├── main.py
│   │   └── utils.py
│   ├── Dockerfile/
│   └── README.md
├── deepstochlog/
│   ├── deepstochlog/
│   ├── workspace/
│   ├── Dockerfile/
│   └── README.md
├── ltntorch/ ...
├── neurasp/ ...
├── pylon/ ...
├── scallop/ ...
├── slash/ ...
├── .gitignore
└── README.md
```

## Riferimenti Bibliografici

1. **Manhaeve, R., Dumančić, S., Kimmig, A., Demeester, T., & De Raedt, L.** (2021).  
   *Neural probabilistic logic programming in DeepProbLog*.  
   Artificial Intelligence.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85104453726&doi=10.1016%2fj.artint.2021.103504&partnerID=40&md5=9b46926a1251f8176c535e6027367e8f) /
   [Link al repo](https://github.com/ML-KULeuven/deepproblog)

2. **Winters, T., Marra, G., Manhaeve, R., & De Raedt, L.** (2022).  
   *DeepStochLog: Neural Stochastic Logic Programming*.  
   Proceedings of the 36th AAAI Conference on Artificial Intelligence, AAAI 2022.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85137088750&doi=10.1609%2faaai.v36i9.21248&partnerID=40&md5=602ba9420d8c16993c33662bd9656d10) /
   [Link al repo](https://github.com/ML-KULeuven/deepstochlog)

3. **Carraro, T.** (2025).  
   *LTNtorch: PyTorch implementation of Logic Tensor Networks*.  
   [Link al paper](https://arxiv.org/pdf/2409.16045) /
   [Link al repo](https://github.com/tommasocarraro/LTNtorch)

4. **Yang, Z., Ishay, A., & Lee, J.** (2020).  
   *NeurASP: Embracing neural networks into answer set programming*.  
   IJCAI International Joint Conference on Artificial Intelligence.  
   [Link al paper](https://arxiv.org/pdf/2307.07700) /
   [Link al repo](https://github.com/azreasoners/NeurASP)

5. **Ahmed, K., Li, T., Ton, T., Guo, Q., Chang, K.-W., Kordjamshidi, P., Srikumar, V., Van den Broeck, G., & Singh, S.** (2022).  
   *PYLON: A PyTorch Framework for Learning with Constraints*.  
   Proceedings of the 36th AAAI Conference on Artificial Intelligence, AAAI 2022.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85127722724&doi=10.1609%2faaai.v36i11.21711&partnerID=40&md5=2a73bf8b9df4878502ad57556bd3710a) /
   [Link al repo](https://github.com/pylon-lib/pylon)

6. **Li, Z., Huang, J., & Naik, M.** (2023).  
   *Scallop: A Language for Neurosymbolic Programming*.  
   Proceedings of the ACM on Programming Languages.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85161998644&doi=10.1145%2f3591280&partnerID=40&md5=4c7ff83ac09ee7dbf4ce0a598e59c29a) /
   [Link al repo](https://github.com/scallop-lang/scallop)

7. **Skryagin, A., Ochs, D., Dhami, D.S., & Kersting, K.** (2023).  
   *Scalable Neural-Probabilistic Answer Set Programming* (SLASH).  
   Journal of Artificial Intelligence Research.  
   [Link al paper](https://www.scopus.com/inward/record.uri?eid=2-s2.0-85178997369&doi=10.1613%2fJAIR.1.15027&partnerID=40&md5=7d949c308cfce51ed99607c371aaab7c) /
   [Link al repo](https://github.com/ml-research/SLASH)


# Autore

## Luca Ferrante
* **Università**: Università di Pisa
* **Corso di Laurea**: Informatica
* **Anno accademico**: 2024/2025