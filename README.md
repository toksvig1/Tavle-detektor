# Tavle-detektor
En kunstig intelligens, som kan ved brug af et neural netværk, detektere farttavler, og påminde føren om fartgrænsen.
Den er bygget fra bunden uden numpy, derfor er netværket også langsommere.

Kør network_trainer.py til at åbne en panel til at træne netværket. Den kan trænes på datasættet i '\dataset'
Programmet gemmer automatisk alle weights og biases i en json fil. Denne fil kan loades i network_runner.py, til at teste netværket.
