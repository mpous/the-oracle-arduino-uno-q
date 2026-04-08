# The Arduino Oracle — Ask the Matrix Oracle on Arduino UNO Q

This project turns your Arduino UNO Q into The Oracle from The Matrix — a mysterious, slightly sarcastic, all-knowing AI that lives entirely on the edge.

To start say “Hey Arduino” to the Arduino UNO Q microphone to wake the Oracle up. Afterwards ask two questions to the Oracle. Receive a cryptic, philosophical answer.

<img width="1624" height="1061" alt="Hey Arduino" src="https://github.com/user-attachments/assets/c1d41b66-95f3-4de8-b44f-db15f8195132" />

Built with Arduino and Edge Impulse (keyword spotting + model cascading) and a small local LLM running via `llama.cpp`. No cloud, no API keys, pure Matrix vibes.

## Project Overview

The Arduino Oracle is a voice-activated, self-hosted fortune teller inspired by the iconic Oracle from The Matrix trilogy.
Using Edge Impulse keyword spotting, it listens for the wake word “Hey Arduino”. Once triggered, it cascades to a lightweight LLM to allow you to ask exactly two questions. 

<img width="1624" height="1061" alt="Meet the Arduino Oracle" src="https://github.com/user-attachments/assets/f5760d9b-772d-44e1-8f79-473fcf4e9179" />

Model cascading perfect for demos!

## Features

* Wake word detection (“Hey Arduino”) using Edge Impulse
* Limited to 2 questions (because even the Oracle has limits)
* Local LLM inference (no Internet required)
* Model cascading for efficient performance
* Fully runs on the Arduino UNO Q


## Hardware Required

* Arduino UNO Q
* USB microphone (for voice input) or USB camera with microphone
* Optional: Edge Impulse account if you want to re-train the model.


## Quick Start

Clone the repository:

```
git clone https://github.com/mpous/the-oracle-arduino-uno-q.git

cd the-oracle-arduino-uno-q
```

Copy the files into the Arduino UNO Q using:

```
scp -r the-oracle-arduino-uno-q/ arduino@<device-ip>:/home/arduino/ArduinoApps/arduino-oracle
```

Go to the Arduino App Lab and you may see the application in the `My apps` section available to run.

Alternatively set up the environment via SSH:

```
ssh arduino@<device-ip>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python download_model.py

python main.py
```

Access the web interface at: http://<your-uno-q-ip>:5001

## How It Works

* **Keyword Spotting**: Edge Impulse model continuously listens for “Hey Arduino”.
* **LLM Response**: Once triggered, it cascades to a small local LLM (SmolLM2 or similar) to generate a Matrix-style answer.

## Disclaimer
Use responsibly. The Oracle is known to be cryptic, slightly judgmental, and occasionally wrong. Predictions are for entertainment purposes only.
Do not use this code in production.

Built with ❤️ from Arduino, Edge Impulse, llama.cpp and the Oracle’ spirit.
