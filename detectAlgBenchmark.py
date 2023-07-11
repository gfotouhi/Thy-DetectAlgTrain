import os, csv, glob
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime


def smooth(x,window_len=10,window='hanning'):

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return np.round(y, decimals = 3)

def consecutiveSum(arr, window_len):
    if arr.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    arrSize = arr.size

    if arrSize < window_len:
        length = arrSize
    length = window_len
    maxSum = np.float64(1.0)
    for i in range(length):
        maxSum += arr[i]
    windowSum = maxSum
    for i in range(length,arrSize):
        windowSum += arr[i] - arr[i - length]
        maxSum = np.maximum(maxSum, windowSum)
    return maxSum

def labelSteps(datas, startPt = 30, rateTh = 0.3, width_LB = 15, avgRate_LB = 0.8):
    
    #if len(datas) >= 10:
    #	datas = smooth(datas)
    dataDiffs = np.diff(datas)

    listOfSteps = []
    inStep = False
    stepL = 0
    stepR = 0
    
    for cnt, diff in enumerate(dataDiffs):
        if cnt < startPt:
            continue
        if not inStep and diff >= rateTh:
            stepL = cnt
            inStep = True
            continue
        if inStep and (diff < rateTh or (cnt == len(dataDiffs) - 1)):
            stepR = cnt
            inStep = False
            LAMPStepFL = False
            stepDiff = 0
            if (stepR - stepL) >= width_LB:
                index = stepL
                while index <= stepR:
                    stepDiff = stepDiff + dataDiffs[index]
                    index += 1
                #print(stepDiff, stepR, stepL)
                avgRate = stepDiff / (stepR - stepL + 1)
                #print(avgRate)
                LAMPStepFL = avgRate >= avgRate_LB
            step = [stepL, stepR, LAMPStepFL]
            stepL = cnt + 1
            listOfSteps.append(step)
            continue
    #print(listOfSteps)
    stepDiff = 0
    cp = 0
    maxDiff = 0
    maxIndex = 0
    stepWidth = 0
    for step in listOfSteps:
        if step[-1]:
            index = step[0] - 1
            stepWidth += step[1] - step[0] + 1

            # Accumulate signal increase of all Ture step as Step Diff
            while index < step[1] + 1:
                stepDiff = stepDiff + dataDiffs[index]
                # Capture time for highest diff as Cp
                if dataDiffs[index] >= maxDiff:
                    maxDiff = dataDiffs[index]
                    maxIndex = index
                index += 1
            if len(datas) > 10: cp = (maxIndex - datas[maxIndex + 1] / dataDiffs[maxIndex]) * 10 / 60 - 5
    avgRate = 0
    if stepWidth != 0: avgRate = stepDiff/stepWidth
    
    return listOfSteps, round(stepDiff, 1), round(cp, 1), round(stepWidth, 1), round(avgRate, 1), round(maxDiff, 1)


def readRunCsv(filename):

    x = []
    signalList = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []

    rlt = []
    idInfo = []
    ChResult = []
    OverallResult = ""

    with open(filename,'r') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        idx = 0
        headerDist = {}

        for row in rows:
            if idx == 0:
                for n, header in enumerate(row):
                    headerDist[header] = n
            if idx == 1:
                idInfo.append([row[0], row[headerDist["Barcode"]]])
                OverallResult = row[headerDist["OverallResult"]]
            if idx == 8:
                x = row[7:]
                x = [float(i)/1000/60 - 5 for i in x]
            if idx == 11:
                ChResult.append(row[5])
                y1 = row[7:]
                rlt.append(row[6])
                y1 = np.array([float(i) for i in y1])
                if len(y1) >= 9: signalList.append(smooth(y1))
            if idx == 12:
                ChResult.append(row[5])
                y2 = row[7:]
                rlt.append(row[6])
                y2 = np.array([float(i) for i in y2])
                if len(y2) >= 9: signalList.append(smooth(y2))
            if idx == 13:
                ChResult.append(row[5])
                y3 = row[7:]
                rlt.append(row[6])
                y3 = np.array([float(i) for i in y3])
                if len(y3) >= 9: signalList.append(smooth(y3))
            if idx == 14:
                ChResult.append(row[5])
                y4 = row[7:]
                rlt.append(row[6])
                y4 = np.array([float(i) for i in y4])
                if len(y4) >= 9: signalList.append(smooth(y4))
            if idx == 15:
                ChResult.append(row[5])
                y5 = row[7:]
                rlt.append(row[6])
                y5 = np.array([float(i) for i in y5])
                if len(y5) >= 9: signalList.append(smooth(y5))

            idx += 1

    return idInfo, OverallResult, signalList


def readTestlog(filename):
    
    testLog = {}
    with open(filename,'r') as csvfile:
        items = csv.reader(csvfile, delimiter=',')
        idx = 0
        for row in items:
            if idx == 0:
                headers = row
                idx += 1
                continue
            inputGroup = row[11]
            testLog[row[0]] = inputGroup
    return testLog

def getMetric():

    dataPath = './data/'
    filenames = sorted(glob.glob(os.path.join(dataPath, '*.csv')))
    testLog = readTestlog('./S2R_testlog.csv')
    # print(testLog)

    idInfo, overallRlt, signalList = "", "", []
    invalidCnt = 0
    truePosCnt = 0
    falsePosCnt = 0
    trueNegCnt = 0
    falseNegCnt = 0

    for filename in filenames:
        idInfo, overallRlt, signalList = readRunCsv(filename)
        sampleGroup = testLog[os.path.splitext(os.path.basename(filename))[0]]

        groundTruth = False
        if sampleGroup != 'Negative':
            groundTruth = True

        startPt, rateTh, width_LB, avgRate_LB = [30, 0.3, 15, 0.8] #rateTh, width_LB, avgRate_LB
        thresholdList = [40, 40, 40, 40, 40]
        rltList = [False, False, False, False, False]

        if len(signalList) != 0:

            for i in range(5):
                _, diff, cp, stepWidth, avgRate, maxDiff= labelSteps(signalList[i], startPt, rateTh, width_LB, avgRate_LB)
                rltList[i] = diff >= thresholdList[i]
        if rltList[0] == False:
            invalidCnt += 1
        else:
            if rltList[1] or rltList[2] or rltList[3] or rltList[4] == True:
                if groundTruth: 
                    truePosCnt += 1
                else:
                    falsePosCnt += 1
            else:
                if groundTruth: 
                    falseNegCnt += 1
                else:
                    trueNegCnt += 1
    cf_matrix = [[truePosCnt, falsePosCnt], [falseNegCnt, trueNegCnt]]
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['True','False'])
    ax.yaxis.set_ticklabels(['True','False'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

if __name__ == '__main__':
    getMetric()

    


