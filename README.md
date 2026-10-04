# satSim

**Simulatore 3D del controllo d'assetto di un CubeSat, con un agente di Reinforcement Learning che impara a orientarlo.**

Un CubeSat 3U nello spazio deve ruotare e puntare con precisione in una direzione data. Lo fa con quattro ruote di reazione: accelerando o frenando un volano, il satellite ruota nel verso opposto.

satSim simula questo sistema in modo realistico e mostra in 3D cosa succede. Confronta inoltre due modi di controllarlo:
- un **controllore classico** (PD), progettato a mano;
- un **agente PPO**, una rete neurale addestrata per prove ed errori che impara da sola a comandare le ruote.

## Cosa si può fare

- **Vedere il satellite in 3D** mentre ruota, con le terne di riferimento, gli assi delle ruote, il Sole e la Terra.
- **Seguire la telemetria in tempo reale**: errore di puntamento, velocità di rotazione, giri delle ruote, consumo elettrico.
- **Disturbare il satellite**: perturbazioni improvvise, spinte esterne, ruote che arrivano al limite di giri.
- **Mettere a confronto l'agente e il controllore classico**, fianco a fianco e partendo dalla stessa situazione.
- **Addestrare nuovi agenti** e valutarli contro il controllore classico.

## Risultati

Media su 10 manovre di prova, partendo da 40–80° dalla direzione voluta:

| | Agente PPO | Controllore PD |
|---|---|---|
| Precisione di puntamento finale | **0.0004°** | 0.0031° |
| Tempo per arrivare entro 1° | **10.4 s** | 12.2 s |
| Energia consumata | 65.8 J | **61.3 J** |

L'agente è circa 8 volte più preciso e più rapido del controllore classico, con un consumo di poco superiore. Il percorso che ha portato a questo risultato è raccontato nel [registro di sviluppo RL](rl/README.md#9-registro-di-sviluppo-rl).

## Avvio rapido

Richiede Python 3.10 o superiore.

```bash
pip install -r requirements.txt

python main.py                        # simulazione 3D con il controllore classico
python main.py --compare --seed 105   # agente PPO (sinistra) contro controllore PD (destra)
python -m rl.evaluate rl/pretrained/local_training_v4_seed1@3_best   # confronto numerico
```

## Com'è fatto

| Cartella | Contenuto |
|---|---|
| `satsim/` | Il simulatore fisico: dinamica del satellite, ruote, disturbi ambientali, controllore PD. |
| `gui/` | L'interfaccia grafica 3D e la finestra di confronto. |
| `rl/` | L'agente: ambiente di addestramento, training, valutazione e modelli già addestrati. |
| `tests/` | Verifiche automatiche della fisica e dell'ambiente RL. |

Il simulatore è indipendente dall'interfaccia: l'agente si addestra senza grafica e molto più veloce del tempo reale.

**Tecnologie:** Python, NumPy, PySide6 + pyqtgraph (grafica 3D), Gymnasium e Stable-Baselines3 (Reinforcement Learning).

## Documentazione

- [Documentazione tecnica](DOCUMENTAZIONE_TECNICA.md): modello fisico ed equazioni, architettura, interfaccia, parametri e verifiche.
- [Reinforcement Learning](rl/README.md): formulazione del problema, reward, addestramento e registro di tutte le versioni dell'agente.
