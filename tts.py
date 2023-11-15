'''
import torch
import os
import sounddevice as sd
import time

language = 'ru'
model_id = 'ru_v3'
sample_rate = 48000 # 48000
speaker = 'kseniya' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu
text = "Привет"

local_file = 'model.pt'

model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                   local_file)

model.to(device)


# воспроизводим
def va_speak(what: str):
    audio = model.apply_tts(text=what+"..",
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo)

    sd.play(audio, sample_rate * 1.05)
    time.sleep((len(audio) / sample_rate) + 0.5)
    sd.stop()

# sd.play(audio, sample_rate)
# time.sleep(len(audio) / sample_rate)
# sd.stop()

'''

from num2words import num2words
import torch
import sounddevice as sd
import time
from googletrans import Translator

print("[!]")

translator = Translator() # Переводчик

# Считывание температуры из файла
f1 = open('weather.txt', 'r')
num = f1.read(1)
num += f1.read(2)
num = num.replace(" ","")
f1.close()
nums = num

# Перевод числа в слово т.к Silero не воспринимает цифры
txt = num2words(num, lang='ru')

# Считывание полного прогноза
f2 = open('weather1.txt', 'r')
text = f2.read()
f2.close()

# Объедение перед озвучиванием
s = txt + " градусов цельсия" + translator.translate(text, dest='ru').text
s.replace(' ', nums)

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

audio = model.apply_tts(text=s,
                        speaker=speaker,
                        sample_rate=sample_rate)

seconds = len(audio) / sample_rate

sd.play(audio, sample_rate)
time.sleep(seconds)
sd.stop()
