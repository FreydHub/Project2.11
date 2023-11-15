import torch
import torchaudio
from omegaconf import OmegaConf
from pprint import pprint
import sounddevice as sd
import time
from googletrans import Translator

language = 'ru'
model_id = 'v4_ru'
device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)  # gpu or cpu
model.speakers
sample_rate = 48000
speaker = 'xenia'
put_accent=True
put_yo=True
example_text = 'В недрах тундры выдры в гетрах тырят в вёдра ядра кедров.'

audio = model.apply_tts(text=example_text,
                        speaker=speaker,
                        sample_rate=sample_rate,
                        put_accent=put_accent,
                        put_yo=put_yo)
print(example_text)
display(Audio(audio, rate=sample_rate))