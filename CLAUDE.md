# satSim — note per Claude Code

Simulatore ADCS di un CubeSat 3U con 4 ruote di reazione, più un agente PPO (Stable-Baselines3) addestrato a sostituire il controllore PD. Dettagli: `README.md` (fisica, architettura, GUI) e `rl/README.md` (logica RL e **registro di sviluppo v1→v8**, da leggere prima di toccare `rl/`).

## Come lavorare con l'utente
- Rispondere in **italiano**.
- **Una modifica alla volta**: soprattutto su reward, osservazione e iperparametri, così l'effetto di ogni modifica resta attribuibile.
- **Prima discutere, poi implementare**: quando l'utente chiede un parere ("dimmi cosa ne pensi"), rispondere senza modificare il codice e aspettare il via.
- Ogni modifica alla logica RL va registrata in `rl/README.md` (sezione "Registro di sviluppo RL"): cosa, perché, risultati.
- Aggiornare questo file quando cambiano stato del progetto, prossimi passi o convenzioni.

## Git
- Si lavora sul branch `claude/vibrant-wozniak-3mn4f8` (o su quello indicato dalla sessione). L'utente verifica e fa il merge su `main` da GitHub; non aprire PR se non richiesto.
- **Non versionare** i risultati dei training: `models/` e `runs/` sono in `.gitignore`. Eccezione: i modelli scelti dall'utente vanno in `rl/pretrained/`.

## Comandi
```bash
pip install -r requirements.txt
python -m pytest -q                                   # 15 test (fisica + RL)
python main.py                                        # GUI con il PD
python main.py --compare --seed 101                   # confronto PPO (sx) vs PD (dx)
python -m rl.train --subproc --out models/<nome>      # 2 M passi, ~45 min nel cloud
python -m rl.evaluate models/<nome>_best              # confronto con il PD sui seed di test
```
- Il training si può lanciare nel cloud: non ha costi di calcolo aggiuntivi oltre all'uso dei messaggi. Lanciarlo in background, salvando l'output in un log nello scratchpad, e riportare in chat solo l'andamento di `ep_rew_mean`, la validazione e la tabella di `rl.evaluate`.
- `--out` evita di sovrascrivere i modelli precedenti; il modello migliore finisce in `<out>_best.zip`.

## Architettura (dipendenze in un solo verso)
- `satsim/`: fisica pura, nessuna dipendenza da Qt o RL. Il limite sulla variazione della coppia (`max_torque_rate`) è nell'engine e vale per ogni controllore.
- `rl/adcs_env.py`: **unica** definizione di osservazione, azione e reward.
  - `rl/train.py`, `rl/evaluate.py` e `rl/compare.py` la riusano, senza duplicare logica RL.
  - Cambiare l'osservazione non richiede modifiche alla GUI, ma rende incompatibili i modelli già addestrati: vanno riaddestrati.
- `gui/`: solo visualizzazione. `theme.py` contiene il tema condiviso; `compare_window.py` importa la parte RL solo con `--compare`.
- Il training non deve mai importare la GUI (lo verifica `test_training_is_headless`).

## Criteri di valutazione
- **Seed di test 100–109** (`rl/evaluate.py`): successo se reward ≥ PD **e** errore finale ≤ 0.01° su tutti i seed.
- **Seed di validazione 200–209**: servono solo a scegliere il modello migliore durante il training, mai per il confronto finale.
- Con lo stesso seed, GUI (`ClosedLoopSimulation`) e ambiente RL partono dalla stessa identica condizione iniziale.
- Quando cambia la reward, ricalcolare la baseline del PD (`python -m rl.evaluate`).

## Stato attuale (2026-10-01)
- **Modello di riferimento: v8** (`rl/pretrained/ppo_adcs_v8.zip`).
  - Osservazione: 10 valori, cioè errore d'assetto in scala log (3), ω (3), coppia corrente (4).
  - Azione: variazione di coppia.
  - Reward: −0.01·θ − 0.02·ln(1+θ/0.01°) − penalità sulle accelerazioni oltre 2 °/s² + bonus sotto 0.1° e 0.01°.
  - Training: learning rate lineare 3e-4→0, 2 M passi.
  - Test: reward −27.8 vs PD −42.9; errore finale 0.0003° vs 0.0031°. **Criteri soddisfatti.**
- **Problema aperto principale: ruote nello spazio nullo.**
  - L'agente accumula il 91–100 % della velocità delle ruote in combinazioni (+,−,+,−) che non agiscono sul corpo, fino alla saturazione (6000 RPM) sui seed 101 e 107. Il PD ne accumula lo 0–9 %.
  - Causa probabile: le velocità delle ruote non sono nell'osservazione né penalizzate nella reward.
  - Prossimo passo da discutere con l'utente: aggiungere Ω/Ω_max all'osservazione e/o una penalità sulla velocità delle ruote.
- **Altri punti aperti** (dopo il precedente):
  - accelerazioni (|α| max 14.1 vs 6.2 °/s²) ed energia (69.1 vs 61.3 J) peggiori del PD;
  - `gamma` 0.99 → 0.995 per i seed con angolo iniziale grande;
  - campionare più spesso le condizioni iniziali difficili, solo se si dimostra che rappresentano una classe;
  - ripetere i training con 2–3 seed per misurarne la variabilità.
