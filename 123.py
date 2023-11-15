import torchaudio
from num2words import num2words
from omegaconf import OmegaConf
import torch
from pprint import pprint
import sounddevice as sd
import time
from googletrans import Translator

print("[!]")

translator = Translator()  # Переводчик


s = 'привет'

# Silero TTS
language = 'ru'
model_id = 'v3_1_ru'
sample_rate = 48000
speaker = 'eugene'
device = torch.device('cpu')
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)

audio = model.apply_tts(text=s, speaker=speaker)  # Дописать метод apply_tts для получения аудио

# Отображение аудио
sd.play(audio, sample_rate)  # Воспроизведение аудио
time.sleep(len(audio) / sample_rate + 1)  # Ожидание окончания аудио
sd.stop()  # Остановка воспроизведения