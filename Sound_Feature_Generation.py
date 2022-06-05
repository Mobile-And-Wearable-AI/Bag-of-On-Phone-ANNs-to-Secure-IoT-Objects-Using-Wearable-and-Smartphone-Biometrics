from librosa import core, feature, effects, get_duration, util
import librosa
import pandas as pd
import numpy as np
import scipy
from spafe.features import gfcc, bfcc, lfcc, mfcc, msrcc, ngcc, pncc, psrcc, lpc, rplp
from scipy.io import wavfile

# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]

def super_append(targetlist, array1d):
    for column in array1d:
        targetlist.append(column)

def repeat_til_len(sound, length):
    output = np.zeros(length)
    repeats = int(length//len(sound))
    pointer = 0
    while(repeats > 0):
        output[pointer:pointer+len(sound)] = sound[:]
        repeats -= 1
        pointer += len(sound)
    output[pointer:] = sound[:len(output[pointer:])]
    return output

def split_file(sound, splits):
    splitsLst = []
    split_len = int(len(sound)//splits)
    pointer = 0
    for i in range(splits):
        split = sound[pointer:pointer+split_len]
        splitsLst.append(split)
        pointer += split_len
    return  splitsLst


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)

FCC = 40
bins = 1024
srValue = 22050*2
sampleRate = srValue
win_hop = .1 #0.01 for splited windows
win_len = .1 #0.01 for splited windows
file_length_to_test = [1.63]  #number of sec #[1.23, 1.33, 1.43, 1.53, 1.63]

p6a, srp6a = core.load("S1.3.12.2020p1.wav", sr = srValue)
p6b, srp6b = core.load("S1.3.12.2020p2.wav", sr = srValue)
p6c, srp6c = core.load("S1.3.12.2020p3.wav", sr = srValue)
p6d, srp6d = core.load("S1.3.12.2020p4.wav", sr = srValue)
p6e, srp6e = core.load("S1.3.12.2020p5.wav", sr = srValue)
p6f, srp6f = core.load("S1.3.12.2020p6.wav", sr = srValue)
p6g, srp6g = core.load("S1.3.12.2020p7.wav", sr = srValue)
p6h, srp6h = core.load("S1.3.12.2020p8.wav", sr = srValue)
p6i, srp6i = core.load("S1.3.12.2020p9.wav", sr = srValue)
p6j, srp6j = core.load("S1.3.12.2020p10.wav", sr = srValue)

playlist6 = [p6a, p6b, p6c, p6d, p6e, p6f]

nVM1, srpVM1 = core.load("1-49098-A-35.wav", sr = srValue)
nVM2, srpVM2 = core.load("1-23996-B-35.wav", sr = srValue)
nVM3, srpVM3 = core.load("1-27165-A-35.wav", sr = srValue)
nVM4, srpVM4 = core.load("1-21896-A-35.wav", sr = srValue)
nVM5, srpVM5 = core.load("1-32373-A-35.wav", sr = srValue)

nWM1, srpWM1 = core.load("1-46744-A-36.wav", sr = srValue)
nWM2, srpWM2 = core.load("1-60460-A-36.wav", sr = srValue)
nWM3, srpWM3 = core.load("1-79236-A-36.wav", sr = srValue)
nWM4, srpWM4 = core.load("2-122820-A-36.wav", sr = srValue)
nWM5, srpWM5 = core.load("1-19840-A-36.wav", sr = srValue)

nAC1, srpAC1 = core.load("35382-0-0-1.wav", sr = srValue)
nAC2, srpAC2 = core.load("54383-0-0-4.wav", sr = srValue)
nAC3, srpAC3 = core.load("58806-0-0-0.wav", sr = srValue)
nAC4, srpAC4 = core.load("60846-0-0-1.wav", sr = srValue)
nAC5, srpAC5 = core.load("73524-0-0-6.wav", sr = srValue)

nDB1, srpDB1 = core.load("344-3-5-0.wav", sr = srValue)
nDB2, srpDB2 = core.load("4918-3-1-0.wav", sr = srValue)
nDB3, srpDB3 = core.load("7383-3-0-0.wav", sr = srValue)
nDB4, srpDB4 = core.load("7913-3-0-0.wav", sr = srValue)
nDB5, srpDB5 = core.load("9031-3-2-0.wav", sr = srValue)

nSR1, srpSR1 = core.load("22601-8-0-0.wav", sr = srValue)
nSR2, srpSR2 = core.load("24347-8-0-11.wav", sr = srValue)
nSR3, srpSR3 = core.load("26173-8-0-0.wav", sr = srValue)
nSR4, srpSR4 = core.load("28426-8-1-0.wav", sr = srValue)
nSR5, srpSR5 = core.load("43805-8-0-0.wav", sr = srValue)


p7a, srp7a = core.load("My recording 1a.wav", sr = srValue)
p7b, srp7b = core.load("My recording 1b.wav", sr = srValue)
p7c, srp7c = core.load("My recording 1c.wav", sr = srValue)
p7d, srp7d = core.load("My recording 1d.wav", sr = srValue)
p7e, srp7e = core.load("My recording 1e.wav", sr = srValue)
p7f, srp7f = core.load("My recording 1f.wav", sr = srValue)
p7g, srp7g = core.load("My recording 1g.wav", sr = srValue)
p7h, srp7h = core.load("My recording 1h.wav", sr = srValue)

playlist7 = [p7a, p7b, p7c, p7d, p7e, p7f, p7g, p7h] #8
playlist7 = [p7a, p7b, p7c, p7d, p7e, p7f]

p8a, srp8a = core.load("S2p1.wav", sr = srValue)
p8b, srp8b = core.load("S2p2.wav", sr = srValue)
p8c, srp8c = core.load("S2p3.wav", sr = srValue)
p8d, srp8d = core.load("S2p4.wav", sr = srValue)
p8e, srp8e = core.load("S2p5.wav", sr = srValue)
p8f, srp8f = core.load("S2p6.wav", sr = srValue)
playlist8 = [p8a, p8b, p8c, p8d, p8e, p8f]

p9a, srp9a = core.load("S3.03.15.2020p1.wav", sr = srValue)
p9b, srp9b = core.load("S3.03.15.2020p2.wav", sr = srValue)
p9c, srp9c = core.load("S3.03.15.2020p3.wav", sr = srValue)
p9d, srp9d = core.load("S3.03.15.2020p4.wav", sr = srValue)
p9e, srp9e = core.load("S3.03.15.2020p5.wav", sr = srValue)
p9f, srp9f = core.load("S3.03.15.2020p6.wav", sr = srValue)
playlist9 = [p9a, p9a, p9c, p9d, p9e, p9f]

p10a, srp10a = core.load("S4.03.15.2020p1.wav", sr = srValue)
p10b, srp10b = core.load("S4.03.15.2020p2.wav", sr = srValue)
p10c, srp10c = core.load("S4.03.15.2020p3.wav", sr = srValue)
p10d, srp10d = core.load("S4.03.15.2020p4.wav", sr = srValue)
p10e, srp10e = core.load("S4.03.15.2020p5.wav", sr = srValue)
p10f, srp10f = core.load("S4.03.15.2020p6.wav", sr = srValue)
playlist10 = [p10a, p10b, p10c, p10d, p10e, p10f]

p11a, srp11a = core.load("S5.03.15.2020p1.wav", sr = srValue)
p11b, srp11b = core.load("S5.03.15.2020p2.wav", sr = srValue)
p11c, srp11c = core.load("S5.03.15.2020p3.wav", sr = srValue)
p11d, srp11d = core.load("S5.03.15.2020p4.wav", sr = srValue)
p11e, srp11e = core.load("S5.03.15.2020p5.wav", sr = srValue)
p11f, srp11f = core.load("S5.03.15.2020p6.wav", sr = srValue)
playlist11 = [p11a, p11b, p11c, p11d, p11e, p11f]

p12a, srp12a = core.load("S6.03.15.2020p1.wav", sr = srValue)
p12b, srp12b = core.load("S6.03.15.2020p2.wav", sr = srValue)
p12c, srp12c = core.load("S6.03.15.2020p3.wav", sr = srValue)
p12d, srp12d = core.load("S6.03.15.2020p4.wav", sr = srValue)
p12e, srp12e = core.load("S6.03.15.2020p5.wav", sr = srValue)
p12f, srp12f = core.load("S6.03.15.2020p6.wav", sr = srValue)
playlist12 = [p12a, p12b, p12c, p12d, p12e, p12f]

p13a, srp13a = core.load("S8p1.wav", sr = srValue)
p13b, srp13b = core.load("S8p2.wav", sr = srValue)
p13c, srp13c = core.load("S8p3.wav", sr = srValue)
p13d, srp13d = core.load("S8p4.wav", sr = srValue)
p13e, srp13e = core.load("S8p5.wav", sr = srValue)
p13f, srp13f = core.load("S8p6.wav", sr = srValue)
playlist13 = [p13a, p13b, p13c, p13d, p13e, p13f]

p14a, srp14a = core.load("S9p1.wav", sr = srValue)
p14b, srp14b = core.load("S9p2.wav", sr = srValue)
p14c, srp14c = core.load("S9p3.wav", sr = srValue)
p14d, srp14d = core.load("S9p4.wav", sr = srValue)
p14e, srp14e = core.load("S9p5.wav", sr = srValue)
p14f, srp14f = core.load("S9p6.wav", sr = srValue)
playlist14 = [p14a, p14b, p14c, p14d, p14e, p14f]

p15a, srp15a = core.load("S10p1.wav", sr = srValue)
p15b, srp15b = core.load("S10p2.wav", sr = srValue)
p15c, srp15c = core.load("S10p3.wav", sr = srValue)
p15d, srp15d = core.load("S10p4.wav", sr = srValue)
p15e, srp15e = core.load("S10p5.wav", sr = srValue)
p15f, srp15f = core.load("S10p6.wav", sr = srValue)
playlist15 = [p15a, p15b, p15c, p15d, p15e, p15f]

lstplaylists =[playlist13, playlist14, playlist15, playlist6, playlist7,
               playlist8, playlist9, playlist10, playlist11, playlist12]

for file_len in file_length_to_test:
    file_array_size = sampleRate * file_len
    sfile = p6a
    #print(sfile, file_array_size)
    sfile = util.fix_length(p6a, int(file_array_size))
    print(get_duration(sfile, sr = sampleRate), " sec")
    print(len(sfile))
    sfilev2 = repeat_til_len(p6a, int(file_array_size))
    print(len(sfilev2))
    print("Hi", len(sfile))

    dfcol_list = ["Person ID", "Sound ID"]


    for bin in range(1,FCC+1):
        dfcol_list.append(str("MFCC") + str(bin))

    print("We have " + str(len(dfcol_list)-2) + " # of features")
    df = pd.DataFrame(None, columns=dfcol_list)
    print(df)
    '''
    hi = [[1,2,3],[4,5,6]]
    print(len(hi))
    print(len(Arrlpc))
    for row in hi:
        print(row)
    ''' # mental test

    print("Computing Pitch shifts")
    #pitchshifts
    pitchshifts= []
    for pitch in range(-7,8):
        pitchshifts.append(pitch/2)

    for lst in range(len(lstplaylists)):
        for sound in range(len(lstplaylists[lst])):
            for i in pitchshifts:
                    sfile = effects.pitch_shift(y=lstplaylists[lst][sound], sr=sampleRate, n_steps=i)
                    sfile = repeat_til_len(sfile, int(file_array_size))
                    datapoint = [lst + 1]
                    datapoint.append(sound + 1)
                    MFCC = np.transpose(mfcc.mfcc(sfile, fs=sampleRate, win_hop=win_hop, win_len=win_len, nfilts=FCC, num_ceps=FCC)).mean(axis=1)
                    datapoint = datapoint + MFCC.tolist()
                    df = df.append(pd.Series(datapoint, index=df.columns), ignore_index=True)
    print("Computing Time shifts")
    #timeshift
    timeshifts= []
    for time in range(1,9):
        if(time/4!=1):
            timeshifts.append(time/4)

    for lst in range(len(lstplaylists)):
        for sound in range(len(lstplaylists[lst])):
            for i in timeshifts:
                sfile = effects.time_stretch(lstplaylists[lst][sound], rate=i)
                datapoint = [lst + 1]
                datapoint.append(sound + 1)
                MFCC = np.transpose(
                    mfcc.mfcc(sfile, fs=sampleRate, win_hop=win_hop, win_len=win_len, nfilts=FCC, num_ceps=FCC)).mean(
                    axis=1)
                datapoint = datapoint + MFCC.tolist()
                df = df.append(pd.Series(datapoint, index=df.columns), ignore_index=True)

    print("Computing Noise")
    #noise
    noisefiles = [nAC1, nAC2, nAC3, nAC4, nAC5, nDB1, nDB2, nDB3, nDB4, nDB5, nSR1, nSR2, nSR3, nSR4, nSR5]
    SNRtargetRange = [.0001, .001, .01, .01, 10, 100, 1000, 10000]

    for lst in range(len(lstplaylists)):
        for sound in range(len(lstplaylists[lst])):
            count = 0
            for i in noisefiles:
                    if (len(lstplaylists[lst][sound]) < len(i)):
                        newsfile = lstplaylists[lst][sound]
                        noise = i[:len(newsfile)]
                    else:
                        noise = i
                        newsfile = lstplaylists[lst][sound]
                        newsfile = newsfile[:len(noise)]

                    filePower = sum(map(lambda i : i * i, lstplaylists[lst][sound]))/len(lstplaylists[lst][sound])
                    noisePower = sum(map(lambda i: i * i, noise))/len(noise)
                    SNRact = filePower/noisePower
                    for SNRtarget in SNRtargetRange:
                        mixedfile = newsfile + ((SNRact / SNRtarget) ** (1 / 2) * noise)
                        sfile = repeat_til_len(mixedfile, int(file_array_size))
                        datapoint = [lst + 1]
                        datapoint.append(sound + 1)
                        MFCC = np.transpose(
                            mfcc.mfcc(sfile, fs=sampleRate, win_hop=win_hop, win_len=win_len, nfilts=FCC,
                                      num_ceps=FCC)).mean(axis=1)
                        datapoint = datapoint + MFCC.tolist()
                        df = df.append(pd.Series(datapoint, index=df.columns), ignore_index=True)
                    count += 1

    dfFile = pd.ExcelWriter("data_outputs\S_10pData_withNoise_MFCC_" + str(file_len) + "sec.xlsx", engine='xlsxwriter')
    print(df)
    df.to_excel(dfFile, sheet_name='Sheet1')
    dfFile.save()
