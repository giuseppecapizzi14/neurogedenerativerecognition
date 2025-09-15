# Progetto ML

Il progetto si basa su modello basato sulle [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
e in particolare sui loro utilizzi e applicazioni per l'analisi di dati audio. La
rete creata può analizzare diversi tipi di dataset audio per compiti di classificazione,
come l'analisi di registrazioni vocali per il riconoscimento di patologie o stati emotivi.

## Requisiti di sistema

Il progetto è stato testato su sistemi con:

- Windows 11:
    - [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) con Ubuntu 22.04.4 LTS (richiede la libreria `libcudart11.0`)
    - CPU: Intel i7-8565u
    - GPU: Nvidia MX250 abilitata per utilizzo capacità di sviluppo CUDA
    - RAM: 16 GB

- MacOs Sonoma:
    - CPU: Chip Apple M2 8-core (4 performance, 4 efficiency)
    - GPU: 8-core
    - Neural Engine: 16-core
    - RAM: 8 GB

## Configurazione pacchetti python e dataset

Il progetto è impostato in modo tale da richiedere solamente l'esecuzione del file di configurazione
[`prepare.sh`](prepare.sh), al cui interno risiedono le istruzioni necessare per scaricare tutti i
pacchetti python utilizzati elencati dentro il file [`requirements.txt`](requirements.txt). Per utilizzare
un dataset specifico, è necessario configurare appropriatamente il percorso nel file di configurazione.

## Configurazione parametri modello

I parametri di configurazione, dentro il file [`base_config.yaml`](config/base_config.yaml) sono
adibiti alla configurazione degli iperparametri del modello, ovvero:

- **data**: parametri di configurazione del dataset

    | Nome               | Tipo         |  Valori accettati  | Descrizione                                                                      |
    | :----------------- | :----------- | :----------------: | :------------------------------------------------------------------------------- |
    | **train_ratio**    | int \| float |       (0, 1)       | proporzione di divisione del dataset in train e test/validazione                 |
    | **test_val_ratio** | int \| float |       (0, 1)       | proporzione di divisione del dataset in test e validazione                       |
    | **data_dir**       | str          | percorso directory | percorso directory dataset (precedentemente scaricati nella directory `dataset`) |

- **model**: parametri di configurazione del modello

    | Nome        | Tipo         | Valori accettati | Descrizione                                              |
    | :---------- | :----------- | :--------------: | :------------------------------------------------------- |
    | **dropout** | int \| float |     \[0, 1]      | percentuale dropout da applicare tra un layer e un altro |

- **training**: parametri di configurazione durante il training

    | Nome                            | Tipo         |                                       Valori accettati                                        | Descrizione                                                                                                                                            |
    | :------------------------------ | :----------- | :-------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
    | **epochs**                      | int          |                                             >= 1                                              | epoche per cui addestrare il modello                                                                                                                   |
    | **batch_size**                  | int          |                                             >= 4                                              | dimensione delle singole batch con cui addestrare il modello                                                                                           |
    | **optimizer**                   | str          | adadelta, adagrad, adamax, adamw, asgd, lbfgs, nadam, radam, rmsprop, rprop, sgd, sparse_adam | ottimizzatore da uilizzare                                                                                                                             |
    | **max_lr**                      | float        |                                              > 0                                              | learning rate da uilizzare come base                                                                                                                   |
    | **min_lr**                      | float        |                                        > 0, <= max_lr                                         | learning rate da uilizzare come minimo per il decay lineare                                                                                            |
    | **warmup_ratio**                | int \| float |                                            \[0, 1]                                            | percentuale di epoche dopo le quali il learning rate andrà a diminuire linearmente                                                                     |
    | **checkpoint_dir**              | str          |                                      percorso directory                                       | percorso directory in cui salvare il modello dopo la fine dell'addestramento                                                                           |
    | **model_name**                  | str          |                                                                                               | nome con cui salvare il modello dopo la fine dell'addestramento                                                                                        |
    | **device**                      | str          |                                        cpu, cuda, mps                                         | dispositivo di accelerazione hardware da utilizzare durante l'addestramento                                                                            |
    | **evaluation_metric**           | str          |                             accuracy, precision, recall, f1, loss                             | metrica da tenere in considerazione durante la valutazione del modello                                                                                 |
    | **best_metric_lower_is_better** | bool         |                                                                                               | indica se la metrica da tenere in considerazione durante la valutazione del modello è da considerarsi migliore se è inferiore o superiore a una soglia |

- parametri singoli:

    | Nome     | Tipo              |           Valori accettati            | Descrizione                                                                                                                               |
    | :------- | :---------------- | :-----------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------- |
    | **plot** | str \| list\[str] | accuracy, precision, recall, f1, loss | metriche da tenere in considerazione durante il plotting delle metriche (questo parametro potrebbe essere omesso per non plottare niente) |

## Risoluzione problemi

Il programma stamperà dei grafici relativi alle metriche richieste dall'utente, quindi nel caso in
cui vengano riscontrati problemi provare ad eseguire il seguente comando (per sistemi basati su
Ubuntu) per installare il package [`tkinter`](https://docs.python.org/3/library/tkinter.html):

```shell
sudo apt-get install python3-tk
```

## Dataset

Il dataset viene gestito dalla classe `AudioDataset` nel file [`dataset.py`](data_classes/dataset.py),
che permette di raccogliere, estrarre e caricare i file audio grezzi da diversi tipi di dataset audio.
La classe supporta diversi formati di dataset e può essere configurata per adattarsi a specifiche
esigenze di classificazione.

### Inizializzazione

La classe richiede i seguenti parametri di inizializzazione:

| Nome          | Tipo | Default | Descrizione                                                                                                   |
| :------------ | :--- | :------ | :------------------------------------------------------------------------------------------------------------ |
| **data_path** | str  |         | il precorso della directory in cui cercare i file audio (precedentemente scaricati nella directory `dataset`) |
| **train**     | bool | True    | se il modello verrà utilizzato durante fase di addestramento o durante la fase di test                        |
| **resample**  | bool | True    | se applicare il resamplig a 16Khz alle tracce audio                                                           |

successivamente eseguirà i seguenti passaggi:

- scansiona le directory contenenti i file audio, e per ogni file audio:
    - estrae le features rilevanti dal percorso del file audio (labels)
    - carica il file audio
    - registra il file audio di lunghezza maggiore
    - immagazzina il percorso del file audio e le sue features rilevanti
- aggiusta il valore della lunghezza del file audio più lungo per tener conto dell'aggiunta di
    padding alle tracce di lunghezza inferiore alla lunghezza massima precedentemente calcolata se
    necessario
- applica un resampling a 16Khz al valore della lunghezza del file audio più lungo se richiesto
    dall'utente

### Estrazione del file audio grezzo

L'estrazione dei file audio e le informazion ad esso relative verranno estratte in maniera lazy
dentro il metodo `__getitem__`, che prendendo come input l'indice del file audio da estrarre,
andrà a:

- recuperare il percorso del file e la label corrispondente
- caricare la waveform e il sample rate associato
- applicare il resampling a 16Khz se richiesto precedentemente e se necessario
- controllare che la traccia sia in formato stereo ed eventualmente duplicare il canale mono nel
    caso contrario
- applicare il padding, sottoforma di audio vuoto (silenzio), alle tracce audio che lo richiedono
    per uniformarsi alla lunghezza massima precedentemente calcolata

Infine ritornerà un'istanza della classe `Sample`, ovvero un dizionario contenente i seguenti campi:

| Nome            | Tipo         | Descrizione                                                                                                                  |
| :-------------- | :----------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **waveform**    | torch.Tensor | il tensore che ci descrive in maniera grezza la forma d'onda del file audio                                                  |
| **sample_rate** | int          | il sample rate della forma d'onda del file audio grezzo                                                                      |
| **label**       | int          | la label relativa alla classe associata al file audio |

## Modello CNN

Il modello che andrà ad analizzare e ad imparare il dataset è basato su un'architettura CNN,
implementata con la classe `CNNModel` secondo l'architettura descritta in
[`cnn_model.py`](model_classes/cnn_model.py).

### Inizializzazione

La classe richiede i seguenti parametri di inizializzazione:

| Dome              | Tipo         | Valori accettati                                               | Descrizione                                                                                   |
| :---------------- | :----------- | :------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| **waveform_size** | int          |                                                                | la dimensione della waveform che andrà analizzata dal modello, ovvero l'input al primo strato |
| **dropout**       | float        | \[0, 1]                                                        | percentuale dropout da applicare tra un layer e un altro                                      |
| **device**        | torch.device | torch.device("cpu"), torch.device("cuda"), torch.device("mps") | dispositivo di accelerazione hardware da utilizzare durante l'addestramento                   |

### Addestramento e test

L'addestramento inizia leggendo i [parametri di configurazione](#configurazione-parametri-modello)
precedenetemente descritti, quindi eseguendo i passaggi in [`train.py`](train.py).

Dopo l'addestramento il modello risultante verrà salvato nella **checkpoint_dir** specificata,
pertanto sarà possibile valutarne e testarne le prestazioni successivamente mediante i passaggi in
[`test.py`](test.py), con i criteri e i metodi di valutazione descritti in [`metrics.py`](metrics.py).
#   T e s t   p u s h   v e l o c e  
 