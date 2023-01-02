import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import Parse

def getNodeGraph(modeList, activityList):
    nodeGraph = [[0 for i in range(len(activityList))] for j in range(len(activityList))]
    for trace in modeList:
        for activityIndex in range(len(trace)):
            if activityIndex == 0:
                nowActivity = trace[activityIndex]
                nowActivityIndex = activityList.index(nowActivity)
            else:
                nowActivity = trace[activityIndex]
                nowActivityIndex = activityList.index(nowActivity)
                lastActivity = trace[activityIndex - 1]
                lastActivityIndex = activityList.index(lastActivity)
                nodeGraph[lastActivityIndex][nowActivityIndex] = 1
    return np.array(nodeGraph)

def getEdgeGraph(graph,activityList):
    edgeList = []
    edgeGraph = [[0 for i in range(len(activityList) ** 2)] for j in range(len(activityList) ** 2)]
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] == 1:
                edgeList.append([i, j])
    for i in range(len(edgeList)):
        for j in range(len(edgeList)):
            if (edgeList[i][1] == edgeList[j][0]):
                m = edgeList[i][0] * len(graph) + edgeList[i][1]
                n = edgeList[j][0] * len(graph) + edgeList[j][1]
                edgeGraph[m][n] = 1
    return np.array(edgeGraph)

def countActivity(logList):
    activityList = []
    activityDic = {}
    for trace in logList:
        for activity in trace:
            if activity in activityList:
                activityDic[activity] += 1
            else:
                activityList.append(activity)
                activityDic[activity] = 1
    return activityList

def frameEmbeding(activityList, frameActivityList):
    featureList = []
    for activity in activityList:
        if activity in frameActivityList:
            index = activityList.index(activity)
            nodeFeature = [0] * (len(activityList))
            nodeFeature[index] = 1
            featureList.append(nodeFeature)
        else:
            nodeFeature = [0] * (len(activityList))
            featureList.append(nodeFeature)
    return torch.Tensor(featureList)

def edgeEmbeding(activityList, graph):
    edgeList = []
    featureList = [[0 for i in range(len(activityList) ** 2)] for j in range(len(activityList) ** 2)]
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] == 1:
                edgeList.append([i, j])
    for i in range(len(edgeList)):
        m = edgeList[i][0] * len(activityList) + edgeList[i][1]
        n = i
        featureList[m][n] = 1
    return torch.Tensor(featureList)

def Embedding(frameLen,stepLen,fileName):
    logList = Parse.Parse(fileName)
    activityList = countActivity(logList)
    activityList.sort()
    startIndex = 0
    splitPoint = startIndex + frameLen
    endIndex = splitPoint + frameLen
    trainSet = {"firstGrah": [], "secondGrah": [], "firstFrameEmbeding": [], "secondFrameEmbeding": [], "firstEdge": [],
                "secondEdge": [], "firstEdgeFeature": [], "secondEdgeFeature": [], "train_y": []}
    while endIndex <= len(logList):
        firstFrame = logList[startIndex:splitPoint]
        secondFrame = logList[splitPoint:endIndex]
        startIndex = startIndex + stepLen
        splitPoint = startIndex + frameLen
        endIndex = splitPoint + frameLen
        firstGrah = getNodeGraph(firstFrame, activityList)
        firstEdge = getEdgeGraph(firstGrah,activityList)
        firstEdgeFeature = edgeEmbeding(activityList, firstGrah)
        secondGrah = getNodeGraph(secondFrame, activityList)
        secondEdge = getEdgeGraph(secondGrah,activityList)
        secondEdgeFeature = edgeEmbeding(activityList, secondGrah)
        firstActivityList = countActivity(firstFrame)
        secondActivityList = countActivity(secondFrame)
        firstFrameEmbeding = frameEmbeding(activityList, firstActivityList)
        secondFrameEmbeding = frameEmbeding(activityList, secondActivityList)
        # train_x = (firstGrah, secondGrah, firstFrameEmbeding, secondFrameEmbeding)
        train_y = 0
        trainSet["firstGrah"].append(firstGrah)
        trainSet["secondGrah"].append(secondGrah)
        trainSet["firstFrameEmbeding"].append(firstFrameEmbeding)
        trainSet["secondFrameEmbeding"].append(secondFrameEmbeding)
        trainSet["firstEdge"].append(firstEdge)
        trainSet["secondEdge"].append(secondEdge)
        trainSet["firstEdgeFeature"].append(firstEdgeFeature)
        trainSet["secondEdgeFeature"].append(secondEdgeFeature)
        trainSet["train_y"].append(train_y)
    return trainSet,activityList

def normalize(A, n):
    A = torch.eye(A.shape[0]) + A
    d = A.sum(1)
    if (n == 1):
        D = torch.diag(torch.pow(d, -0.5))
    else:
        D = torch.diag_embed(torch.pow(d, -0.5))
    return torch.matmul(torch.matmul(D, A), D)

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, activityListLen):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in, bias=False)
    def forward(self, X, A):
        X = F.relu(self.fc1(X.mm(A)))
        X = F.relu(self.fc2(X.mm(A)))
        X = torch.sum(X, dim=0)
        X = F.relu(X)
        return X

def euclidean_distance(x, y):
    return torch.sum((x - y) ** 2, dim=-1)

def createTest(activityList,trainSet,frameLen,stepLen):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 0:
        print('Lets use', torch.cuda.device_count(), 'GPUs!')
    gcn1 = GCN(len(activityList),len(activityList),activityList)
    gcn2 = GCN(len(activityList)**2,len(activityList)**2,activityList)
    gcn1.to(device)
    gcn2.to(device)
    gcn_optimizer1 = torch.optim.Adam(gcn1.parameters(), lr=0.001)
    gcn_optimizer2 = torch.optim.Adam(gcn2.parameters(), lr=0.001)
    disList1 = []
    disList2 = []
    SPList = []
    SP = frameLen
    stepLen = stepLen
    for i in range(len(trainSet["train_y"])):
        SPList.append(SP)
        SP = SP + stepLen
        firstGrah = trainSet["firstGrah"][i]
        secondGrah = trainSet["secondGrah"][i]
        firstFrameEmbeding = trainSet["firstFrameEmbeding"][i]
        secondFrameEmbeding = trainSet["secondFrameEmbeding"][i]
        firstEdge = trainSet["firstEdge"][i]
        secondEdge = trainSet["secondEdge"][i]
        firstEdgeFeature = trainSet["firstEdgeFeature"][i]
        secondEdgeFeature = trainSet["secondEdgeFeature"][i]
        y = trainSet["train_y"][i]
        y = torch.Tensor([y])
        y = y.to(device)
        firstA = normalize(firstGrah, 1)
        secondA = normalize(secondGrah, 1)
        firstA = firstA.to(torch.float32)
        secondA = secondA.to(torch.float32)

        firstEdgeA = normalize(firstEdge, 1)
        secondEdgeA = normalize(secondEdge, 1)
        firstEdgeA = firstEdgeA.to(torch.float32)
        secondEdgeA = secondEdgeA.to(torch.float32)

        firstFrameEmbeding = firstFrameEmbeding.to(device)
        firstA = firstA.to(device)
        secondFrameEmbeding = secondFrameEmbeding.to(device)
        secondA = secondA.to(device)
        firstEdgeFeature = firstEdgeFeature.to(device)
        firstEdgeA = firstEdgeA.to(device)
        secondEdgeFeature = secondEdgeFeature.to(device)
        secondEdgeA = secondEdgeA.to(device)

        f1 = gcn1(firstFrameEmbeding, firstA)
        f2 = gcn1(secondFrameEmbeding, secondA)
        f3 = gcn2(firstEdgeFeature, firstEdgeA)
        f4 = gcn2(secondEdgeFeature, secondEdgeA)

        distance1 = euclidean_distance(f1, f2)
        distance2 = euclidean_distance(f3, f4)
        disList1.append(float(distance1))
        disList2.append(float(distance2))
    return sumDis(disList1,disList2,SPList)

def sumDis(disList1,disList2,SPList):
    dislist1_max=max(disList1)
    dislist1_min=min(disList1)
    disListG1=[]
    for i in disList1:
        new_dis1=(i-dislist1_min)/(dislist1_max-dislist1_min)
        disListG1.append(float(new_dis1))

    dislist2_max=max(disList2)
    dislist2_min=min(disList2)
    disListG2=[]
    for j in disList2:
        if (j - dislist2_min < 1 ** -10):
            new_dis2 = 0
        else:
            new_dis2=(j-dislist2_min)/(dislist2_max-dislist2_min)
        disListG2.append(float(new_dis2))
    sumDisList=[]
    a=0.5
    for i in range(len(disList1)):
        sumDisList.append(float(a*disListG1[i]+(1-a)*disListG2[i]))
    return sumDisList,SPList

def kmeans(sumDisList,SPList,jump_dis,change_num):
    xArray = np.array(sumDisList,dtype = float)
    a = xArray.reshape(-1,1)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(a)
    y_kmeans = kmeans.predict(a)
    num=0
    y_kmeans=y_kmeans.tolist()
    pos0=y_kmeans.index(0)
    pos1=y_kmeans.index(1)
    if sumDisList[pos0]>sumDisList[pos1]:
        num=0
    elif sumDisList[pos1]>sumDisList[pos0]:
        num=1
    sum_change=findChange(num,y_kmeans,jump_dis,SPList)
    avg_list=filterChange(sum_change,change_num)
    return avg_list

def findChange(num,y_kmeans,jump,SPList):
    sum_change=[]
    change_list10=[]
    if(num==0):
        un_num=1
    else:
        un_num=0
    sign=un_num
    now_jump=0
    for i in range(len(y_kmeans)):
        if y_kmeans[i]!=num and sign!=num:
            continue
        elif y_kmeans[i]!=num and sign==num:
            now_jump=now_jump+1
            if now_jump>=jump:
                sum_change.append(change_list10)
                change_list10=[]
                sign=un_num
                now_jump=0
        elif y_kmeans[i]==num and sign!=num:
            change_list10.append(SPList[i])
            sign=num
        elif y_kmeans[i]==num and sign==num:
            now_jump=0
            change_list10.append(SPList[i])
        if i == len(y_kmeans) - 1 and sign == num:
            sum_change.append(change_list10)
    return sum_change

def filterChange(sum_change,change_num):
    avg_list=[]
    for i in sum_change:
        if len(i)>=change_num:
            avg_pos=(i[0]+i[len(i)-1])/2
            avg_list.append(int(avg_pos))
    return avg_list

def get_data(filename):
    filename=os.path.basename(filename)
    number_first=list(filter(str.isdigit,filename))[0]
    index1=filename.find(number_first)
    index2=filename.find('.')
    num=filename[index1:index2]
    name=filename[:index2]
    return num,name

def demo_plot(x, y, x_maxsize, title, x_major_locator,num,filename):
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(dpi=1080)
    plt.plot(x, y, linewidth=2.0, color="deepskyblue")
    plt.title(title)
    plt.xlabel("trace", fontsize=15)
    plt.ylabel("distance", fontsize=15)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    maxsize = x_maxsize
    m = 0.2
    N = len(x)
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    if num == "2.5k":
        sign = 250
    elif num == "5k":
        sign = 500
    elif num == "7.5k":
        sign = 750
    elif num == "10k":
        sign = 1000
    for i in range(9):
        plt.axvline((i + 1) * sign, color='red', linestyle='--')

    plt.savefig("%s%s.jpg" % (os.path.dirname(filename), title), bbox_inches='tight', dpi=1080)
    #plt.show()
    plt.close()

def main():
    usage = """\
    usage:
        driftDetection.py [-w value] [-r value] [-p value] log_file_path
    options:
        -w complete window size, integer, default value is 100
        -j detection window size, integer, default value is 3
        -n stable period, integer, default value is 3
        """
    import getopt, sys

    try:
        opts, args = getopt.getopt(sys.argv[1:], "w:j:n:")
        if len(args) == 0:
            print(usage)
            return

        window_size = 100
        jump_dis = 3
        change_num = 3
        for opt, value in opts:
            if opt == '-w':
                window_size = int(value)
            elif opt == '-j':
                jump_dis = int(value)
            elif opt == '-n':
                change_num = int(value)

        print("--------------------------------------------------------------")
        print(" Log: ", args[0])
        print(" window_size: ", window_size)
        print(" jump_dis: ", jump_dis)
        print(" change_num: ", change_num)
        print("--------------------------------------------------------------")

        trainSet,activityList=Embedding(window_size,10,args[0])
        sumDisList,SPList=createTest(activityList,trainSet,window_size,10)
        avg_list=kmeans(sumDisList,SPList,jump_dis,change_num)
        print("All change points detected: ", avg_list)
        num,name=get_data(args[0])
        demo_plot(SPList, sumDisList,30,name,250,num,args[0])
    except getopt.GetoptError:
        print(usage)
    except SyntaxError as error:
        print(error)
        print(usage)
    return 0


if __name__ == '__main__':
    main()
