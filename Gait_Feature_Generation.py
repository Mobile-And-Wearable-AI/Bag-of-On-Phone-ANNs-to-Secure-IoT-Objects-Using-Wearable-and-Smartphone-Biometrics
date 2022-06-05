import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

df = pd.read_excel("Gait 1600-1609 watch data.xlsx", "Sheet1", header=0)
dfTemp = pd.DataFrame(None, columns=list(df.columns.values))
df2col_list = ["Person ID",
"X-acc mean", " X-acc median", " X-acc sdv", " X-acc variance", " X-acc coef of variance", " X-acc range", " X-acc coef of range", " X-acc Q1", " X-acc Q3", " X-acc Max", " X-acc interquartile range", " X-acc coef of interquartile", " X-acc mean absolute deviation", " X-acc median absolute deviation", " X-acc energy", " X-acc power", " X-acc RMS", " X-acc RSS", " X-acc SNR", " X-acc skewness", " X-acc kurtosis",
               "Y-acc mean", " Y-acc median", " Y-acc sdv", " Y-acc variance", " Y-acc coef of variance", " Y-acc range", " Y-acc coef of range", " Y-acc Q1", " Y-acc Q3", " Y-acc Max", " Y-acc interquartile range", " Y-acc coef of interquartile", " Y-acc mean absolute deviation", " Y-acc median absolute deviation", " Y-acc energy", " Y-acc power", " Y-acc RMS", " Y-acc RSS", " Y-acc SNR", " Y-acc skewness", " Y-acc kurtosis",
               "Z-acc mean", " Z-acc median", " Z-acc sdv", " Z-acc variance", " Z-acc coef of variance", " Z-acc range", " Z-acc coef of range", " Z-acc Q1", " Z-acc Q3", " Z-acc Max", " Z-acc interquartile range", " Z-acc coef of interquartile", " Z-acc mean absolute deviation", " Z-acc median absolute deviation", " Z-acc energy", " Z-acc power", " Z-acc RMS", " Z-acc RSS", " Z-acc SNR", " Z-acc skewness", " Z-acc kurtosis",
               "X-gy mean", " X-gy median", " X-gy sdv", " X-gy variance", " X-gy coef of variance", " X-gy range", " X-gy coef of range", " X-gy Q1", " X-gy Q3", " X-gy Max", " X-gy interquartile range", " X-gy coef of interquartile", " X-gy mean absolute deviation", " X-gy median absolute deviation", " X-gy energy", " X-gy power", " X-gy RMS", " X-gy RSS", " X-gy SNR", " X-gy skewness", " X-gy kurtosis",
               "Y-gy mean", " Y-gy median", " Y-gy sdv", " Y-gy variance", " Y-gy coef of variance", " Y-gy range", " Y-gy coef of range", " Y-gy Q1", " Y-gy Q3", " Y-gy Max", " Y-gy interquartile range", " Y-gy coef of interquartile", " Y-gy mean absolute deviation", " Y-gy median absolute deviation", " Y-gy energy", " Y-gy power", " Y-gy RMS", " Y-gy RSS", " Y-gy SNR", " Y-gy skewness", " Y-gy kurtosis",
               "Z-gy mean", " Z-gy median", " Z-gy sdv", " Z-gy variance", " Z-gy coef of variance", " Z-gy range", " Z-gy coef of range", " Z-gy Q1", " Z-gy Q3", " Z-gy Max", " Z-gy interquartile range", " Z-gy coef of interquartile", " Z-gy mean absolute deviation", " Z-gy median absolute deviation", " Z-gy energy", " Z-gy power", " Z-gy RMS", " Z-gy RSS", " Z-gy SNR", " Z-gy skewness", " Z-gy kurtosis",
               "acc mean", "acc median", "acc sdv", "acc variance", "acc coef of variance", "acc range", "acc coef of range", "acc Q1", "acc Q3", "acc Max", "acc interquartile range", "acc coef of interquartile", "acc mean absolute deviation", "acc median absolute deviation", "acc energy", "acc power", "acc RMS", "acc RSS", "acc SNR", "acc skewness", "acc kurtosis",
               "gy mean", "gy median", "gy sdv", "gy variance", "gy coef of variance", "gy range", "gy coef of range", "gy Q1", "gy Q3", "gy Max", "gy interquartile range", "gy coef of interquartile", "gy mean absolute deviation", "gy median absolute deviation", "gy energy", "gy power", "gy RMS", "gy RSS", "gy SNR", "gy skewness", "gy kurtosis"
               ]

df2 = pd.DataFrame(None, columns=df2col_list)

def fusion(x,y,z):
    return np.sqrt(x**2+y**2+z**2)


j = 0
snipLen = 10
while j < len(df["Person ID"]):

    dfTemp = df.loc[j:j + snipLen - 1]
    if(len(dfTemp)!= snipLen):
        break
    else:
        meanID = dfTemp["Person ID"].mean()

        acc = fusion(dfTemp["X-acc"], dfTemp["Y-acc"], dfTemp["Z-acc"])
        gy = fusion(dfTemp["X-gy"], dfTemp["Y-gy"], dfTemp["Z-gy"])

        meanXacc = dfTemp["X-acc"].mean()
        meanYacc = dfTemp["Y-acc"].mean()
        meanZacc = dfTemp["Z-acc"].mean()
        meanXgy = dfTemp["X-gy"].mean()
        meanYgy = dfTemp["Y-gy"].mean()
        meanZgy = dfTemp["Z-gy"].mean()
        meanacc, meangy = acc.mean(), gy.mean()

        medianXacc = dfTemp["X-acc"].median()
        medianYacc = dfTemp["Y-acc"].median()
        medianZacc = dfTemp["Z-acc"].median()
        medianXgy = dfTemp["X-gy"].median()
        medianYgy = dfTemp["Y-gy"].median()
        medianZgy = dfTemp["Z-gy"].median()
        medianacc, mediangy = acc.median(), gy.median()

        varXacc = dfTemp["X-acc"].var()
        varYacc = dfTemp["Y-acc"].var()
        varZacc = dfTemp["Z-acc"].var()
        varXgy = dfTemp["X-gy"].var()
        varYgy = dfTemp["Y-gy"].var()
        varZgy = dfTemp["Z-gy"].var()
        varacc, vargy = acc.var(), gy.var()

        stdXacc = varXacc ** (1 / 2)
        stdYacc = varXacc ** (1 / 2)
        stdZacc = varXacc ** (1 / 2)
        stdXgy = varXacc ** (1 / 2)
        stdYgy = varXacc ** (1 / 2)
        stdZgy = varXacc ** (1 / 2)
        stdacc, stdgy = varacc ** (1 / 2), vargy ** (1 / 2)

        coOfVarXacc = (stdXacc / meanXacc) * 100
        coOfVarYacc = (stdYacc / meanYacc) * 100
        coOfVarZacc = (stdZacc / meanZacc) * 100
        coOfVarXgy = (stdXgy / meanXgy) * 100
        coOfVarYgy = (stdYgy / meanYgy) * 100
        coOfVarZgy = (stdZgy / meanZgy) * 100
        coOfVaracc, coOfVargy = (stdacc / meanacc) * 100, (stdgy / meangy) * 100

        rngeXacc = dfTemp["X-acc"].max() - dfTemp["X-acc"].min()
        rngeYacc = dfTemp["Y-acc"].max() - dfTemp["Y-acc"].min()
        rngeZacc = dfTemp["Z-acc"].max() - dfTemp["Z-acc"].min()
        rngeXgy = dfTemp["X-gy"].max() - dfTemp["X-gy"].min()
        rngeYgy = dfTemp["Y-gy"].max() - dfTemp["Y-gy"].min()
        rngeZgy = dfTemp["Z-gy"].max() - dfTemp["Z-gy"].min()
        rngeacc, rngegy = acc.max() - acc.min(), gy.max() - gy.min()

        if (dfTemp["X-acc"].max() + dfTemp["X-acc"].min() == 0):
            corngeXacc = 0
        else:
            corngeXacc = rngeXacc / (dfTemp["X-acc"].max() + dfTemp["X-acc"].min())
        if (dfTemp["Y-acc"].max() + dfTemp["Y-acc"].min() == 0):
            corngeYacc = 0
        else:
            corngeYacc = rngeYacc / (dfTemp["Y-acc"].max() + dfTemp["Y-acc"].min())
        if (dfTemp["Z-acc"].max() + dfTemp["Z-acc"].min() == 0):
            corngeZacc = 0
        else:
            corngeZacc = rngeZacc / (dfTemp["Z-acc"].max() + dfTemp["Z-acc"].min())
        if (dfTemp["X-gy"].max() + dfTemp["X-gy"].min() == 0):
            corngeXgy = 0
        else:
            corngeXgy = rngeXgy / (dfTemp["X-gy"].max() + dfTemp["X-gy"].min())
        if (dfTemp["Y-gy"].max() + dfTemp["Y-gy"].min() == 0):
            corngeYgy = 0
        else:
            corngeYgy = rngeYgy / (dfTemp["Y-gy"].max() + dfTemp["Y-gy"].min())
        if (dfTemp["Z-gy"].max() + dfTemp["Z-gy"].min() == 0):
            corngeZgy = 0
        else:
            corngeZgy = rngeZgy / (dfTemp["Z-gy"].max() + dfTemp["Z-gy"].min())
        if (acc.max() + acc.min() == 0):
            corngeacc = 0
        else:
            corngeacc = rngeacc / (acc.max() + acc.min())
        if (gy.max() + gy.min() == 0):
            corngegy = 0
        else:
            corngegy = rngegy / (gy.max() + gy.min())

        q1Xacc, q3Xacc = dfTemp["X-acc"].quantile([.25, .75])
        q1Yacc, q3Yacc = dfTemp["Y-acc"].quantile([.25, .75])
        q1Zacc, q3Zacc = dfTemp["Z-acc"].quantile([.25, .75])
        q1Xgy, q3Xgy = dfTemp["X-gy"].quantile([.25, .75])
        q1Ygy, q3Ygy = dfTemp["Y-gy"].quantile([.25, .75])
        q1Zgy, q3Zgy = dfTemp["Z-gy"].quantile([.25, .75])
        q1acc, q3acc = acc.quantile([.25, .75])
        q1gy, q3gy = gy.quantile([.25, .75])

        maxXacc = dfTemp["X-acc"].max()
        maxYacc = dfTemp["Y-acc"].max()
        maxZacc = dfTemp["Z-acc"].max()
        maxXgy = dfTemp["X-gy"].max()
        maxYgy = dfTemp["Y-gy"].max()
        maxZgy = dfTemp["Z-gy"].max()
        maxacc, maxgy = acc.max(), gy.max()

        #minacc, mingy = acc.min(), gy.min()

        iqrXacc = q3Xacc - q1Xacc
        iqrYacc = q3Yacc - q1Yacc
        iqrZacc = q3Zacc - q1Zacc
        iqrXgy = q3Xgy - q1Xgy
        iqrYgy = q3Ygy - q1Ygy
        iqrZgy = q3Zgy - q1Zgy
        iqracc, iqrgy = q3acc - q1acc, q3gy - q1gy

        coiXacc = iqrXacc / 2
        coiYacc = iqrYacc / 2
        coiZacc = iqrZacc / 2
        coiXgy = iqrXgy / 2
        coiYgy = iqrYgy / 2
        coiZgy = iqrZgy / 2
        coiacc, coigy = iqracc / 2, iqrgy / 2

        absumMeanDifXacc = 0
        absumMeanDifYacc = 0
        absumMeanDifZacc = 0
        absumMeanDifXgy = 0
        absumMeanDifYgy = 0
        absumMeanDifZgy = 0
        for i in range(snipLen):
            absumMeanDifXacc = absumMeanDifXacc + abs((dfTemp["X-acc"][i + j] - meanXacc))
            absumMeanDifYacc = absumMeanDifYacc + abs((dfTemp["Y-acc"][i + j] - meanYacc))
            absumMeanDifZacc = absumMeanDifZacc + abs((dfTemp["Z-acc"][i + j] - meanZacc))
            absumMeanDifXgy = absumMeanDifXgy + abs((dfTemp["X-gy"][i + j] - meanXgy))
            absumMeanDifYgy = absumMeanDifYgy + abs((dfTemp["Y-gy"][i + j] - meanYgy))
            absumMeanDifZgy = absumMeanDifZgy + abs((dfTemp["Z-gy"][i + j] - meanZgy))
        mad_meanXacc = absumMeanDifXacc / snipLen
        mad_meanYacc = absumMeanDifYacc / snipLen
        mad_meanZacc = absumMeanDifZacc / snipLen
        mad_meanXgy = absumMeanDifXgy / snipLen
        mad_meanYgy = absumMeanDifYgy / snipLen
        mad_meanZgy = absumMeanDifZgy / snipLen
        absumMeanDifacc, absumMeanDifgy = 0, 0
        for i in range(snipLen):
            absumMeanDifacc = absumMeanDifacc + abs((acc[i+j] - meanacc))
            absumMeanDifgy = absumMeanDifgy + abs((gy[i+j] - meangy))
        mad_meanacc, mad_meangy = absumMeanDifacc / snipLen, absumMeanDifgy / snipLen

        normdfTempXacc = abs(dfTemp["X-acc"] - medianXacc)
        normdfTempYacc = abs(dfTemp["Y-acc"] - medianYacc)
        normdfTempZacc = abs(dfTemp["Z-acc"] - medianZacc)
        normdfTempXgy = abs(dfTemp["X-gy"] - medianXgy)
        normdfTempYgy = abs(dfTemp["Y-gy"] - medianYgy)
        normdfTempZgy = abs(dfTemp["Z-gy"] - medianZgy)
        mad_medianXacc = normdfTempXacc.median()
        mad_medianYacc = normdfTempYacc.median()
        mad_medianZacc = normdfTempZacc.median()
        mad_medianXgy = normdfTempXgy.median()
        mad_medianYgy = normdfTempYgy.median()
        mad_medianZgy = normdfTempZgy.median()
        normdfTempacc, normdfTempgy = abs(acc - medianacc), abs(gy - mediangy)
        mad_medianacc, mad_mediangy = normdfTempacc.median(), normdfTempgy.median()

        energyXacc = 0
        energyYacc = 0
        energyZacc = 0
        energyXgy = 0
        energyYgy = 0
        energyZgy = 0
        for i in range(snipLen):
            energyXacc = energyXacc + dfTemp["X-acc"][i + j] ** 2
            energyYacc = energyYacc + dfTemp["Y-acc"][i + j] ** 2
            energyZacc = energyZacc + dfTemp["Z-acc"][i + j] ** 2
            energyXgy = energyXgy + dfTemp["X-gy"][i + j] ** 2
            energyYgy = energyYgy + dfTemp["Y-gy"][i + j] ** 2
            energyZgy = energyZgy + dfTemp["Z-gy"][i + j] ** 2
        energyacc, energygy = 0, 0
        for i in range(snipLen):
            energyacc, energygy = energyacc + acc[i+j] ** 2, energygy + gy[i+j] ** 2

        powerXacc = energyXacc / snipLen
        powerYacc = energyYacc / snipLen
        powerZacc = energyZacc / snipLen
        powerXgy = energyXgy / snipLen
        powerYgy = energyYgy / snipLen
        powerZgy = energyZgy / snipLen
        poweracc, powergy = energyacc / snipLen, energygy / snipLen

        rootMeanSquXacc = powerXacc ** (1 / 2)
        rootMeanSquYacc = powerYacc ** (1 / 2)
        rootMeanSquZacc = powerZacc ** (1 / 2)
        rootMeanSquXgy = powerXgy ** (1 / 2)
        rootMeanSquYgy = powerYgy ** (1 / 2)
        rootMeanSquZgy = powerZgy ** (1 / 2)
        rootMeanSquacc, rootMeanSqugy = poweracc ** (1 / 2), powergy **(1 / 2)

        rootSumSquXacc = energyXacc ** (1 / 2)
        rootSumSquYacc = energyYacc ** (1 / 2)
        rootSumSquZacc = energyZacc ** (1 / 2)
        rootSumSquXgy = energyXgy ** (1 / 2)
        rootSumSquYgy = energyYgy ** (1 / 2)
        rootSumSquZgy = energyZgy ** (1 / 2)
        rootSumSquacc, rootSumSqugy = energyacc ** (1 / 2), energygy ** (1 / 2)

        if (stdXacc == 0):
            sigNoiseRatioXacc = 0
        else:
            sigNoiseRatioXacc = meanXacc / stdXacc
        if (stdYacc == 0):
            sigNoiseRatioYacc = 0
        else:
            sigNoiseRatioYacc = meanYacc / stdYacc
        if (stdZacc == 0):
            sigNoiseRatioZacc = 0
        else:
            sigNoiseRatioZacc = meanZacc / stdZacc
        if (stdXgy == 0):
            sigNoiseRatioXgy = 0
        else:
            sigNoiseRatioXgy = meanXgy / stdXgy
        if (stdYgy == 0):
            sigNoiseRatioYgy = 0
        else:
            sigNoiseRatioYgy = meanYgy / stdYgy
        if (stdZgy == 0):
            sigNoiseRatioZgy = 0
        else:
            sigNoiseRatioZgy = meanZgy / stdZgy
        if (stdacc == 0):
            sigNoiseRatioacc = 0
        else:
            sigNoiseRatioacc = meanacc / stdacc
        if (stdgy == 0):
            sigNoiseRatiogy = 0
        else:
            sigNoiseRatiogy = meangy / stdgy

        sumCubedXacc = 0
        sumCubedYacc = 0
        sumCubedZacc = 0
        sumCubedXgy = 0
        sumCubedYgy = 0
        sumCubedZgy = 0
        for i in range(snipLen):
            sumCubedXacc = sumCubedXacc + dfTemp["X-acc"][i + j] ** 3
            sumCubedYacc = sumCubedYacc + dfTemp["Y-acc"][i + j] ** 3
            sumCubedZacc = sumCubedZacc + dfTemp["Z-acc"][i + j] ** 3
            sumCubedXgy = sumCubedXgy + dfTemp["X-gy"][i + j] ** 3
            sumCubedYgy = sumCubedYgy + dfTemp["Y-gy"][i + j] ** 3
            sumCubedZgy = sumCubedZgy + dfTemp["Z-gy"][i + j] ** 3
        if (stdXacc == 0):
            skewnessXacc = 0
        else:
            skewnessXacc = sumCubedXacc / ((snipLen - 1) * stdXacc ** 3)
        if (stdYacc == 0):
            skewnessYacc = 0
        else:
            skewnessYacc = sumCubedYacc / ((snipLen - 1) * stdYacc ** 3)
        if (stdZacc == 0):
            skewnessZacc = 0
        else:
            skewnessZacc = sumCubedZacc / ((snipLen - 1) * stdZacc ** 3)

        if (stdXgy == 0):
            skewnessXgy = 0
        else:
            skewnessXgy = sumCubedXgy / ((snipLen - 1) * stdXgy ** 3)
        if (stdYgy == 0):
            skewnessYgy = 0
        else:
            skewnessYgy = sumCubedYgy / ((snipLen - 1) * stdYgy ** 3)
        if (stdZgy == 0):
            skewnessZgy = 0
        else:
            skewnessZgy = sumCubedZgy / ((snipLen - 1) * stdZgy ** 3)
        sumCubedacc, sumCubedgy = 0, 0
        for i in range(snipLen):
            sumCubedacc = sumCubedacc + acc[i+j] ** 3
            sumCubedgy = sumCubedgy + gy[i+j] ** 3
        if(stdacc == 0):
            skewnessacc = 0
        else:
            skewnessacc = sumCubedacc / ((snipLen - 1) * stdacc ** 3)
        if(stdgy == 0):
            skewnessgy = 0
        else:
            skewnessgy = sumCubedgy / ((snipLen - 1) * stdgy ** 3)

        sumQuartXacc = 0
        sumQuartYacc = 0
        sumQuartZacc = 0
        sumQuartXgy = 0
        sumQuartYgy = 0
        sumQuartZgy = 0
        for i in range(snipLen):
            sumQuartXacc = sumQuartXacc + dfTemp["X-acc"][i + j] ** 4
            sumQuartYacc = sumQuartYacc + dfTemp["Y-acc"][i + j] ** 4
            sumQuartZacc = sumQuartZacc + dfTemp["Z-acc"][i + j] ** 4
            sumQuartXgy = sumQuartXgy + dfTemp["X-gy"][i + j] ** 4
            sumQuartYgy = sumQuartYgy + dfTemp["Y-gy"][i + j] ** 4
            sumQuartZgy = sumQuartZgy + dfTemp["Z-gy"][i + j] ** 4
        if (stdXacc == 0):
            kurtosisXacc = 0
        else:
            kurtosisXacc = sumQuartXacc / ((snipLen - 1) * stdXacc ** 4)
        if (stdYacc == 0):
            kurtosisYacc = 0
        else:
            kurtosisYacc = sumQuartYacc / ((snipLen - 1) * stdYacc ** 4)
        if (stdZacc == 0):
            kurtosisZacc = 0
        else:
            kurtosisZacc = sumQuartZacc / ((snipLen - 1) * stdZacc ** 4)
        if (stdXgy == 0):
            kurtosisXgy = 0
        else:
            kurtosisXgy = sumQuartXgy / ((snipLen - 1) * stdXgy ** 4)
        if (stdYgy == 0):
            kurtosisYgy = 0
        else:
            kurtosisYgy = sumQuartYgy / ((snipLen - 1) * stdYgy ** 4)
        if (stdZgy == 0):
            kurtosisZgy = 0
        else:
            kurtosisZgy = sumQuartZgy / ((snipLen - 1) * stdZgy ** 4)
        sumQuartacc, sumQuartgy = 0, 0
        for i in range(snipLen):
            sumQuartacc = sumQuartacc + acc[i+j] ** 4
            sumQuartgy = sumQuartgy + gy[i+j] ** 4
        if(stdacc == 0):
            kurtosisacc = 0
        else:
            kurtosisacc = sumQuartacc / ((snipLen - 1) * stdacc ** 4)
        if(stdgy == 0):
            kurtosisgy = 0
        else:
            kurtosisgy = sumQuartgy / ((snipLen - 1) * stdgy ** 4)

            df2 = df2.append(
                pd.Series([int(meanID),
                           meanXacc, medianXacc, stdXacc, varXacc, coOfVarXacc, rngeXacc, corngeXacc, q1Xacc, q3Xacc,
                           maxXacc, iqrXacc, coiXacc, mad_meanXacc, mad_medianXacc, energyXacc, powerXacc,
                           rootMeanSquXacc, rootSumSquXacc, sigNoiseRatioXacc, skewnessXacc, kurtosisXacc,
                           meanYacc, medianYacc, stdYacc, varYacc, coOfVarYacc, rngeYacc, corngeYacc, q1Yacc, q3Yacc,
                           maxYacc, iqrYacc, coiYacc, mad_meanYacc, mad_medianYacc, energyYacc, powerYacc,
                           rootMeanSquYacc, rootSumSquYacc, sigNoiseRatioYacc, skewnessYacc, kurtosisYacc,
                           meanZacc, medianZacc, stdZacc, varZacc, coOfVarZacc, rngeZacc, corngeZacc, q1Zacc, q3Zacc,
                           maxZacc, iqrZacc, coiZacc, mad_meanZacc, mad_medianZacc, energyZacc, powerZacc,
                           rootMeanSquZacc, rootSumSquZacc, sigNoiseRatioZacc, skewnessZacc, kurtosisZacc,
                           meanXgy, medianXgy, stdXgy, varXgy, coOfVarXgy, rngeXgy, corngeXgy, q1Xgy, q3Xgy, maxXgy,
                           iqrXgy, coiXgy, mad_meanXgy, mad_medianXgy, energyXgy, powerXgy, rootMeanSquXgy,
                           rootSumSquXgy, sigNoiseRatioXgy, skewnessXgy, kurtosisXgy,
                           meanYgy, medianYgy, stdYgy, varYgy, coOfVarYgy, rngeYgy, corngeYgy, q1Ygy, q3Ygy, maxYgy,
                           iqrYgy, coiYgy, mad_meanYgy, mad_medianYgy, energyYgy, powerYgy, rootMeanSquYgy,
                           rootSumSquYgy, sigNoiseRatioYgy, skewnessYgy, kurtosisYgy,
                           meanZgy, medianZgy, stdZgy, varZgy, coOfVarZgy, rngeZgy, corngeZgy, q1Zgy, q3Zgy, maxZgy,
                           iqrZgy, coiZgy, mad_meanZgy, mad_medianZgy, energyZgy, powerZgy, rootMeanSquZgy,
                           rootSumSquZgy, sigNoiseRatioZgy, skewnessZgy, kurtosisZgy,
                           meanacc, medianacc, stdacc, varacc, coOfVaracc, rngeacc, corngeacc, q1acc, q3acc,
                           maxacc, iqracc, coiacc, mad_meanacc, mad_medianacc, energyacc, poweracc,
                           rootMeanSquacc, rootSumSquacc, sigNoiseRatioacc, skewnessacc, kurtosisacc,
                           meangy, mediangy, stdgy, vargy, coOfVargy, rngegy, corngegy, q1gy, q3gy, maxgy,
                           iqrgy, coigy, mad_meangy, mad_mediangy, energygy, powergy, rootMeanSqugy,
                           rootSumSqugy, sigNoiseRatiogy, skewnessgy, kurtosisgy,
                           ], index=df2.columns),
                ignore_index=True)

        j += int(snipLen/2)

dfFile = pd.ExcelWriter("data_outputs\Gait_10P_data.xlsx", engine='xlsxwriter')
print(df2)
df2.to_excel(dfFile,sheet_name='Sheet1')
dfFile.save()