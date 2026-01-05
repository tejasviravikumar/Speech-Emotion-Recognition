import pandas as pd
import numpy as np
import librosa,glob,os

emotionCodes = {
    "01" : "neutral",
    "02" : "calm",
    "03" : "happy",
    "04" : "sad",
    "05" : "angry",
    "06" : "fearful",
    "07" : "disgust",
    "08" : "surprised"
}

"""
Each audio file is named as seven two-digit numbers separated by dashes: AA-BB-CC-DD-EE-FF-GG.wav
        
AA = modality      (each file is marked 03 for audio-only)
BB = vocal channel (each file is marked 01 for speech)
CC = emotion       (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
DD = intensity     (01 = normal, 02 = strong)
EE = statement     (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
FF = repetition    (01 = 1st repetition, 02 = 2nd repetition)
GG = actor         (01 to 24. Odd numbered actors are male, even numbered actors are female)
"""

def extract_features(audio):
    y , sr = librosa.load(audio)

    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=30)
    mfcc_mean = np.mean(mfcc,axis=1)

    spec_contrat = librosa.feature.spectral_contrast(y=y,sr=sr)
    spec_contrast_mean = np.mean(spec_contrat,axis=1)

    chroma = librosa.feature.chroma_stft(y=y,sr=sr)
    chroma_mean = np.mean(chroma,axis=1)

    filename = os.path.basename(audio)
    feat = filename.split('.')[0].split('-')
    modality = feat[0] 
    vocal_channel = feat[1] 
    emotion_code = feat[2] 
    emotion = emotionCodes.get(emotion_code) 
    emotional_intensity = feat[3] 
    statement = feat[4] 
    repetition = feat[5] 
    actor_id = feat[6] 
    gender = "Male" if int(actor_id) % 2 == 1 else "Female"

    return {
        "Filename": filename,
        "Modality": modality, 
        "Actor_ID": actor_id, 
        "Gender": gender, 
        "Emotional_Intensity": emotional_intensity,
        "StatementNo": statement, 
        "RepetitionNo": repetition, 
        "Vocal_Channel": vocal_channel,
        **{f"mfcc_{i+1}": mfcc_mean[i] for i in range(30)},
        **{f"sc_{i+1}": spec_contrast_mean[i] for i in range(7)},
        **{f"chroma_{i+1}": chroma_mean[i] for i in range(12)},
        "Emotion": emotion
    }

audio_files = glob.glob("./Ravedess_dataset/*/*.wav")
all_data = []
for file in audio_files:
    all_data.append(extract_features(file))


df = pd.DataFrame(all_data)
df.to_csv("ravdess_metadata.csv",index=False)