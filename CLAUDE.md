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
python -m pytest -q                                   # 18 test (fisica + RL)
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
  - Cambiare l'osservazione non richiede modifiche alla GUI, ma rende incompatibili i modelli già addestrati. Per mantenerli utilizzabili si aggiunge un'opzione all'ambiente (come `wheel_speed_obs`) e la si gestisce in `rl.models.env_kwargs_for`, che sceglie il formato in base al modello.
- `gui/`: solo visualizzazione. `theme.py` contiene il tema condiviso; `compare_window.py` importa la parte RL solo con `--compare`.
- Il training non deve mai importare la GUI (lo verifica `test_training_is_headless`).
- Modelli: caricarli sempre con `rl.models.load_model`, mai con `PPO.load`. Non mettere funzioni Python (closure, lambda) negli iperparametri: verrebbero salvate come bytecode, non portabile tra versioni di Python (lo verifica `test_saved_models_are_portable`). Per gli schedule usare le classi di Stable-Baselines3.

## Ambiente locale dell'utente
- Fedora, Python 3.14, ROS 2 installato (`~/ros2_lyrical`). Il `source` di ROS 2 imposta `PYTHONPATH` e `LD_LIBRARY_PATH`, che interferiscono con satSim: plugin pytest di ROS, Qt di sistema caricata al posto di quella di PySide6.
- Consigliato: virtualenv `.venv` in un terminale senza variabili di ROS (`unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH`) e PyTorch solo CPU (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
- Il cloud usa Python 3.11: i modelli addestrati qui devono funzionare anche con 3.14.

## Criteri di valutazione
- **Seed di test 100–109** (`rl/evaluate.py`): successo se reward ≥ PD **e** errore finale ≤ 0.01° su tutti i seed.
- **Seed di validazione 200–209**: servono solo a scegliere il modello migliore durante il training, mai per il confronto finale.
- Con lo stesso seed, GUI (`ClosedLoopSimulation`) e ambiente RL partono dalla stessa identica condizione iniziale.
- Quando cambia la reward, ricalcolare la baseline del PD (`python -m rl.evaluate`).

## Stato attuale (2026-10-02)
- **Modello di riferimento: v8** (`rl/pretrained/ppo_adcs_v8.zip`).
  - Osservazione: 10 valori, cioè errore d'assetto in scala log (3), ω (3), coppia corrente (4).
  - Azione: variazione di coppia.
  - Reward: −0.01·θ − 0.02·ln(1+θ/0.01°) − penalità sulle accelerazioni oltre 2 °/s² + bonus sotto 0.1° e 0.01°.
  - Training: learning rate lineare 3e-4→0, 2 M passi.
  - Test: reward −27.8 vs PD −42.9; errore finale 0.0003° vs 0.0031°. **Criteri soddisfatti.**
- **Codice: v9.** L'osservazione include Ω/Ω_max delle ruote (14 valori); la reward è invariata rispetto alla v8. Modello v9 non ancora addestrato: l'utente lo addestra in locale.
- **Problema aperto principale: ruote nello spazio nullo.**
  - L'agente accumula il 91–100 % della velocità delle ruote in combinazioni (+,−,+,−) che non agiscono sul corpo, fino alla saturazione (6000 RPM) sui seed 101 e 107. Il PD ne accumula lo 0–9 %.
  - Causa probabile: le velocità delle ruote non sono nell'osservazione né penalizzate nella reward.
  - v9 aggiunge Ω/Ω_max all'osservazione: da verificare con il training se basta.
  - Passo successivo previsto: una reward che penalizzi la velocità delle ruote. La forma è da decidere con l'utente; ridurre la velocità dovrebbe abbassare anche i consumi, ma per l'energia l'utente pensa di agire anche sull'accelerazione.
  - Dopo le verifiche l'utente porterà tutto su `main` e si proseguirà su un branch nuovo.
- **Altri punti aperti** (dopo il precedente):
  - accelerazioni (|α| max 14.1 vs 6.2 °/s²) ed energia (69.1 vs 61.3 J) peggiori del PD;
  - `gamma` 0.99 → 0.995 per i seed con angolo iniziale grande;
  - campionare più spesso le condizioni iniziali difficili, solo se si dimostra che rappresentano una classe;
  - ripetere i training con 2–3 seed per misurarne la variabilità.
