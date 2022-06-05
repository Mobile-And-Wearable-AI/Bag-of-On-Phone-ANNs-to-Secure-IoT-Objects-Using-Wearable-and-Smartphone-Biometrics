import pandas as pd

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


df = pd.read_excel("NetHealth_HR_Activity_10p.xlsx", "Sheet1", header=0)
dfTemp = pd.DataFrame(None, columns=list(df.columns.values))
df2col_list = ["Person ID", "mean", "median", "sdv", "variance", "coef of variance", "range", "coef of range", "Q1",
               "Q3", "Max", "interquartile range", "coef of interquartile", "mean absolute deviation", "median absolute deviation","energy","power", "RMS", "RSS",
               "SNR","skewness", "kurtosis"]

df2 = pd.DataFrame(None, columns=df2col_list)



j = 0
snipLen = 10
while j < len(df["Person ID"]):

    dfTemp = df.loc[j:j + snipLen - 1]
    if (len(dfTemp) != snipLen):
        break
    else:

        meanID = dfTemp["Person ID"].mean()
        meanHR = dfTemp["Heart Rate"].mean()
        median = dfTemp["Heart Rate"].median()
        variance = dfTemp["Heart Rate"].var()
        std = variance ** (1 / 2)
        coOfVar = (std / meanHR) * 100
        rnge = dfTemp["Heart Rate"].max() - dfTemp["Heart Rate"].min()
        cornge = rnge / (dfTemp["Heart Rate"].max() + dfTemp["Heart Rate"].min())
        q1, q3 = dfTemp["Heart Rate"].quantile([.25, .75])
        max = dfTemp["Heart Rate"].max()
        iqr = q3 - q1
        coi = iqr / 2

        absumMeanDif = 0
        for i in range(snipLen):
            absumMeanDif = absumMeanDif + abs((dfTemp["Heart Rate"][i + j] - meanHR))
        mad_mean = absumMeanDif / snipLen

        normdfTemp = abs(dfTemp["Heart Rate"] - median)
        mad_median = normdfTemp.median()

        energy = 0
        for i in range(snipLen):
            energy = energy + dfTemp["Heart Rate"][i + j]**2

        power = energy/snipLen

        rootMeanSqu = power**(1/2)

        rootSumSqu = energy**(1/2)

        if(std == 0):
            sigNoiseRatio = 0
        else:
            sigNoiseRatio = meanHR/std

        sumCubed= 0
        for i in range(snipLen):
            sumCubed = sumCubed + dfTemp["Heart Rate"][i + j]**3

        if(std == 0):
            skewness = 0
        else:
            skewness = sumCubed/((snipLen-1)*std**3)

        sumQuart = 0
        for i in range(snipLen):
            sumQuart = sumQuart + dfTemp["Heart Rate"][i + j] ** 4

        if(std == 0):
            kurtosis = 0
        else:
            kurtosis = sumQuart / ((snipLen - 1) * std ** 4)

        df2 = df2.append(
            pd.Series([int(meanID), meanHR, median, std, variance, coOfVar, rnge, cornge,
                       q1, q3, max, iqr, coi, mad_mean, mad_median, energy, power, rootMeanSqu,
                       rootSumSqu,sigNoiseRatio, skewness, kurtosis], index=df2.columns),
            ignore_index=True)
        for i in range(snipLen):
            dfTemp = dfTemp.drop(df.index[i + j], axis=0)
        j += int(snipLen/2)

dfFile = pd.ExcelWriter("data_outputs\HR_10P_data.xlsx", engine='xlsxwriter')

df2.to_excel(dfFile,sheet_name='Sheet1')
dfFile.save()
