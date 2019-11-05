
import csv 
from ripser import ripser, lower_star_img
from persim import plot_diagrams
import numpy as np
from sklearn import datasets
import pandas as pd
import numpy as np
import random as ran
import scipy
from scipy import ndimage
import PIL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
#**************
# ************
#  **********
#   ********
#    ******
# ************
#  **********
#   ********
#    ******
#     ****
#      **
TESTFILE= 'TestA.csv'

ListOfCharacter=[]
with open('letters.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    
    for row in reader:
        DataFromCSVLine = row[0].split(',')[1:]
        tenBytenMatrix = []
        for RowsInMatrix in range (10):
            tenBytenMatrix.append(DataFromCSVLine[RowsInMatrix*10:(RowsInMatrix+1)*10])
        
        ListOfCharacter.append(tenBytenMatrix)
        
#Below is the 10x10 matrix being matched to a variable matching the letter which the data represents
A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,Char1,Char2,Char3,Char4,Char5,Char6= (lett for lett in ListOfCharacter)
s=[A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,Char1,Char2,Char3,Char4,Char5,Char6]    
t=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Period','Comma','Space','Dash','Colon','Semicolon']
j=0
ClassificationKey=[]
dataset=[]

def ClassifyH1(i):
    
    ranges,ClassificationKey= RipserMat(i,j)      
   
    ranges=list(ranges)
    dataset.append(ranges)    

    return ClassificationKey
    
    

def Diag_Array_LL(matrix): #lower left corner diagnal 
    AlteredMatrix =  np.zeros((10,10))
    
    BelowDiagonal = []
    AboveDiagonal =[]
    for diag_value in range(-9,10):
        if diag_value < 1:
            BelowDiagonal.append(np.diag(matrix,k=diag_value))
        else:
            AboveDiagonal.append(np.diag(matrix,k=diag_value))
            
    for array in BelowDiagonal:
        row = 0
        for column in range(10-len(array),10):
            if int(array[row]) == 0:
                AlteredMatrix[column,row] = 100
            else:
                AlteredMatrix[column,row] = int(array[row])*len(array)
            row +=1

    for array in AboveDiagonal:
        row = 0
        for column in range(10-len(array),10):
            if int(array[row]) == 0:
                AlteredMatrix[row,column] = 100
            else:
                AlteredMatrix[row,column] = int(array[row])*((10-len(array))+10)
            row +=1
  
    return AlteredMatrix

def CenterInFiltration(array):
    AlteredMatrix =  np.zeros((10,10))
    startingSquare = 4
    endingSquare = 5
    mainHolder= []
    SquareValueHolder = []
    filteringArray1 = []
    filteringArray2 = []
                               
                               
    for numberofsquares in range(1,6):
        for x in range(startingSquare,endingSquare+1):
            for y in range(startingSquare,endingSquare+1):
                SquareValueHolder.append([x,y])
        
        startingSquare -=1
        endingSquare += 1

        if numberofsquares == 1:
            mainHolder.append(SquareValueHolder)

        elif numberofsquares == 2:
            mainHolder.append([i for i in SquareValueHolder if i not in mainHolder[0]])
            
        elif numberofsquares > 2:
            for arrays in mainHolder:
                SquareValueHolder = [i for i in SquareValueHolder if i not in arrays]
            mainHolder.append(SquareValueHolder)
            
        SquareValueHolder = []

    multiplier = 1 
    for element in reversed(mainHolder):
        for locations in reversed(element):
            if int(array[int(locations[0])][int(locations[1])]) * (multiplier) == 0:
                AlteredMatrix[int(locations[0])][int(locations[1])] = 100
            else:
                AlteredMatrix[int(locations[0])][int(locations[1])] = int(array[int(locations[0])][int(locations[1])]) * (multiplier)
        multiplier += 1 

    return (AlteredMatrix)

def CenterOutFiltration(array):
    AlteredMatrix =  np.zeros((10,10))
    startingSquare = 4
    endingSquare = 5
    mainHolder= []
    SquareValueHolder = []
    filteringArray1 = []
    filteringArray2 = []
                               
                               
    for numberofsquares in range(1,6):
        for x in range(startingSquare,endingSquare+1):
            for y in range(startingSquare,endingSquare+1):
                SquareValueHolder.append([x,y])
        
        startingSquare -=1
        endingSquare += 1

        if numberofsquares == 1:
            mainHolder.append(SquareValueHolder)

        elif numberofsquares == 2:
            mainHolder.append([i for i in SquareValueHolder if i not in mainHolder[0]])
            
        elif numberofsquares > 2:
            for arrays in mainHolder:
                SquareValueHolder = [i for i in SquareValueHolder if i not in arrays]
            mainHolder.append(SquareValueHolder)
            
        SquareValueHolder = []

    multiplier = 1 
    for element in mainHolder:
        for locations in element:
            if int(array[int(locations[0])][int(locations[1])]) * (multiplier) == 0:
                AlteredMatrix[int(locations[0])][int(locations[1])] = 100
            else:
                AlteredMatrix[int(locations[0])][int(locations[1])] = int(array[int(locations[0])][int(locations[1])]) * (multiplier)
        multiplier += 1 

    return (AlteredMatrix)    
        

    
def LeftToRight(matrix):
       
    for column in range(10):
        
        for row in range(10):
            if matrix[column][row] == 0.0:
                matrix[column][row]=100.0
               
            else:
                matrix[column][row]=float(row) +1.0
                
        
    return matrix
  
def RightToLeft(matrix):
       
    for row in range(10):        
        for column in range(10):
            if matrix[column][row] == 0.0:
                matrix[column][row]=100.0
               
            else:
                matrix[column][row]=10.0 -float(row) 
            
        
    return matrix

def BottomToTop(matrix):
       
    for row in range(10):        
        for column in range(10):
            if matrix[column][row] == 0.0:
                matrix[column][row]=100.0
               
            else:
                matrix[column][row]=10.0 -float(column) 
           

    return matrix

def TopToBottom(matrix):
       
    for column in range(10):
        
        for row in range(10):
            if matrix[column][row] == 0.0:
                matrix[column][row]=100.0
               
            else:
                matrix[column][row]=float(column) +1.0
       
    return matrix

def USDictionaryUpdate(dgm):
    if len(dgm) == 3:
        if dgm[0][1]==float('inf'):
            RA=(dgm[0][0]),(100.0)
        else:
            RA=(dgm[0][0]),(dgm[0][1])
        
        if dgm[1][1]==float('inf'):
            RB=(dgm[1][0]),(100.0)
        else:
            RB=(dgm[1][0]),(dgm[1][1])
            
        if dgm[2][1]==float('inf'):
            RC=(dgm[2][0]),(100.0)
        else:
            RC=(dgm[2][0]),(dgm[2][1])
        
        
    elif len(dgm) == 2:
        if dgm[0][1]==float('inf'):
            RA=(dgm[0][0]),(100.0)
        else:
            RA=(dgm[0][0]),(dgm[0][1])
        
        if dgm[1][1]==float('inf'):
            RB=(dgm[1][0]),(100.0)
        else:
            RB=(dgm[1][0]),(dgm[1][1])
        RC=(0,0)
    elif len(dgm) == 1:
        if dgm[0][1]==float('inf'):
            RA=(dgm[0][0]),(100.0)
        else:
            RA=(dgm[0][0]),(dgm[0][1])
        
        RB=(0,0)
        RC=(0,0)
    else:
        RA=(0,0)
        RB=(0,0)
        RC=(0,0)
    LSA= RA[1]-RA[0]
    LSB= RB[1]-RB[0]
    LSC= RC[1]-RC[0]
    return(LSA,LSB,LSC)
    
def RipserMat(i,j):
    index=[]
    if j!=99:
        j=s.index(i)
    else:
        j=99          
    mat=np.array(i) #prodce single matrix per letter
    mat=mat.astype(np.float) #convert to float
    co = np.argwhere(mat!=0)# create matrix of coordinates 
    if len(co)==0: #if empty data set then all values are 0
        H1=0;H2=0;H3=0;R1=(0,0);R2=(0,0);RL0=0;RRL0=(0,0);RRL1=(0,0);RRL2=(0,0);LR0=0;RLR0=(0,0);RLR1=(0,0);RLR2=(0,0);TB0=0;RTB0=(0,0);RTB1=(0,0);RTB2=(0,0);
        BT0=0;RBT0=(0,0);RBT1=(0,0);RBT2=(0,0);LL0=0;LL1= 0;LL2=0;LLLS0=0;LLLS1=0;LLLS2=0;CO0=0;CO1= 0;CO2=0;COLS0=0;COLS1=0;COLS2=0;CI0=0;
        CI1= 0;CI2=0;CILS0=0;CILS1=0;CILS2=0;RLLS0= 0;RLLS1=0;RLLS2=0;LRLS0=0;LRLS1=0;LRLS2=0;TBLS0=0;TBLS1=0;TBLS2=0;BTLS0=0;BTLS1=0;BTLS2=0
        ranges= [RLLS0,RLLS1,RLLS2,LRLS0,LRLS1,LRLS2,TBLS0,TBLS1,TBLS2,BTLS0,BTLS1,BTLS2,LLLS0,LLLS1,LLLS2,COLS0,COLS1,COLS2,CILS0,CILS1,CILS2,H1,H2,H3]  

    else:
        
        dgms = ripser(co)['dgms']#analyses the point cloud of coordinates in ripser 
        OneDset=(dgms[1:][0])#defines the range of 1D homology ""

        if len(OneDset)>0: #delete all repeated values
            OneDset=np.unique(OneDset, axis=0)          

        H1=len(OneDset)
        
        for h in np.arange(len(OneDset)):
            index1=(OneDset[h][1])
            index0=(OneDset[h][0])
            feat=np.subtract((OneDset[h][1]),(OneDset[h][0]))
            
            #removes noise in 1D
            if feat < 1:
                index.append(h)
                                    
        OneDset= np.delete(OneDset,index,0)
                      
        H1=(len(OneDset))
        H2=0
        H3=0
        
        if H1 ==1:
            H1=0
            H2=100
            H3=0
        elif H1 ==0 :
            H1=100
        elif H1 ==2:
            H1=0
            H2=0
            H3=100            
    
            #RIGHT TO LEFT scan    
        RLmat=RightToLeft(mat) 
        dgm = lower_star_img(RLmat)
        RL0=len(dgm)
        RLLS0,RLLS1,RLLS2= USDictionaryUpdate(dgm)
        mat=np.array(i) 
        mat=mat.astype(np.float)
            
        LRmat=LeftToRight(mat) 
        dgm = lower_star_img(LRmat)
        LR0=len(dgm)
        LRLS0,LRLS1,LRLS2= USDictionaryUpdate(dgm)
        mat=np.array(i) 
        mat=mat.astype(np.float)
            
        TBmat=TopToBottom(mat) 
            
        dgm = lower_star_img(TBmat)
        TB0=len(dgm)
        TBLS0,TBLS1,TBLS2= USDictionaryUpdate(dgm)
        mat=np.array(i) 
        mat=mat.astype(np.float)
            
        BTmat=BottomToTop(mat) 
        dgm = lower_star_img(BTmat)
        BT0=len(dgm)
        BTLS0,BTLS1,BTLS2= USDictionaryUpdate(dgm)
        mat=np.array(i) 
        mat=mat.astype(np.float)
            
        LLmat= Diag_Array_LL(mat)
        dgm = lower_star_img(LLmat)
        LL0=len(dgm)
        LLLS0,LLLS1,LLLS2= USDictionaryUpdate(dgm)
        mat=np.array(i) 
        mat=mat.astype(np.float)
            
            
        COmat=CenterOutFiltration(mat)
        dgm = lower_star_img(COmat)
        CO0=len(dgm)
        COLS0,COLS1,COLS2= USDictionaryUpdate(dgm)
        mat=np.array(i) 
        mat=mat.astype(np.float)
            
        CImat=CenterInFiltration(mat)
        dgm = lower_star_img(CImat)
        CI0=len(dgm)
        CILS0,CILS1,CILS2= USDictionaryUpdate(dgm)
        mat=np.array(i) 
        mat=mat.astype(np.float)
            
        ranges= [RLLS0,RLLS1,RLLS2,LRLS0,LRLS1,LRLS2,TBLS0,TBLS1,TBLS2,BTLS0,BTLS1,BTLS2,LLLS0,LLLS1,LLLS2,COLS0,COLS1,COLS2,CILS0,CILS1,CILS2,H1,H2,H3]  
    
    for i in range(len(ranges)):
        if i<=2 and ranges[i]!=0.0:
            ClassificationKey.append([j,1,ranges[i]])
        elif 2<i<=5 and ranges[i]!=0.0:
            ClassificationKey.append([j,2,ranges[i]])
        elif 5<i<=8 and ranges[i]!=0.0:
            ClassificationKey.append([j,3,ranges[i]])
        elif 8<i<=11 and ranges[i]!=0.0:
            ClassificationKey.append([j,4,ranges[i]])
        elif 11<i<=14 and ranges[i]!=0.0:
            ClassificationKey.append([j,5,ranges[i]])
        elif 14<i<=17 and ranges[i]!=0.0:
            ClassificationKey.append([j,6,ranges[i]])
        elif 17<i<=20 and ranges[i]!=0.0:
            ClassificationKey.append([j,7,ranges[i]]) 
            
    return ranges,ClassificationKey


def UnknownLetter(file, j):
   
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        
        for row in reader:
            csvdata = row[0].split(',')[0:]
            unknown = []
            for RowsInMat in range (10):
                unknown.append(csvdata[RowsInMat*10:(RowsInMat+1)*10])
            print(np.array(unknown))
    UnknownV,UnknownT=RipserMat(unknown,j)
    
    #find distance between all vectors with 
    a=np.array(UnknownV)   
    Distances=[]
    Hamm=[]
    Mink=[]
    Distperc=[]
    Hammperc=[]
    Minkperc=[]
     
    for index, row in ClassificationOutput.iterrows():
        b=np.array(row)     
        dist=distance.euclidean(a,b)  
        hamm=distance.hamming(a,b)
        mink=distance.minkowski(a,b,1)
        manh=distance.cityblock(a,b)
        Distances.append(dist)
        Hamm.append(hamm)
        Mink.append(mink)
        
    maxDist= np.max(Distances)
    maxMink= np.max(Mink)
    minDist= np.min(Distances)
    minHamm= np.min(Hamm)
    minMink= np.min(Mink)  
    
    for i in Distances:
        perc=(1-(i-minDist)/( maxDist-minDist))*100
        Distperc.append(perc)
    for i in Hamm:
        perc=(1- ((i/1)))*100
        Hammperc.append(perc)
    for i in Mink:
        perc= (1-(i-minMink)/( maxMink-minMink))*100
        Minkperc.append(perc)

    Distperc=np.array(Distperc)
    Distperc=np.around(Distperc,decimals=4)
    Distperc=np.sort(Distperc)
 
    Hammperc=np.array(Hammperc)
    Hammperc=np.around(Hammperc,decimals=4)
    Hammperc=np.sort(Hammperc)
    
    Minkperc=np.array(Minkperc)
    Minkperc=np.around(Minkperc,decimals=4)
    Minkperc=np.sort(Minkperc)
    
    Answer=Distances.index(min(Distances))
    Distances[Answer]=1000
    Answer=t[Answer:Answer+1]

    Answer2=Distances.index(min(Distances))
    Distances[Answer2]=1000
    Answer2=t[Answer2:Answer2+1]    
    Answer3=Distances.index(min(Distances))
    Answer3=t[Answer3:Answer3+1]
    
    Hammanswer=Hamm.index(min(Hamm))
    Hamm[Hammanswer]=Hamm[Hammanswer]+10
    Hammanswer=t[Hammanswer:Hammanswer+1]
    Hammanswer2=Hamm.index(min(Hamm))
    Hamm[Hammanswer2]=Hamm[Hammanswer2]+10
    Hammanswer2=t[Hammanswer2:Hammanswer2+1]    
    Hammanswer3=Hamm.index(min(Hamm))
    Hammanswer3=t[Hammanswer3:Hammanswer3+1]
    
    Minkanswer=Mink.index(min(Mink))
    Mink[Minkanswer]=1000
    Minkanswer=t[Minkanswer:Minkanswer+1]
    Minkanswer2=Mink.index(min(Mink))
    Mink[Minkanswer2]=1000
    Minkanswer2=t[Minkanswer2:Minkanswer2+1]    
    Minkanswer3=Mink.index(min(Mink))
    Minkanswer3=t[Minkanswer3:Minkanswer3+1]
    

    
    return UnknownV,UnknownT,Distperc,Hammperc,Minkperc,Distances,Hamm, Mink, Answer,Answer2,Answer3,Hammanswer,Hammanswer2,Hammanswer3,Minkanswer,Minkanswer2,Minkanswer3
            

            
            
ClassificationKey=[ClassifyH1(i) for i in s]


ClassificationOutput=pd.DataFrame(dataset,columns=["RLLS0","RLLS1","RLLS2", "LRLS0","LRLS1","LRLS2","TBLS0",'TBLS1','TBLS2','BTLS0','BTLS1','BTLS2','LLLS0','LLLS1','LLLS2','COLS0','COLS1','COLS2','CILS0','CILS1','CILS2',"H1","H2", "H3"])
DuplicatedFeatureMat=ClassificationOutput[ClassificationOutput.duplicated()]

UnknownV,UnknownT,Distperc,Hammperc,Minkperc,Distances,Hamm, Mink,Answer,Answer2,Answer3,Hammanswer,Hammanswer2,Hammanswer3,Minkanswer,Minkanswer2,Minkanswer3=UnknownLetter(TESTFILE,99) 
Results=[Answer,Answer2,Answer3,Hammanswer,Hammanswer2,Hammanswer3,Minkanswer,Minkanswer2,Minkanswer3]
ClassificationKey=np.array(ClassificationKey)
print('\n')
print('Eucladian Distance: ' +str(Answer) +',' +str(Answer2) +',' +str(Answer3))
print (np.flip(Distperc[np.argsort(Distperc)[-3:]]))
print('\n')
print('Hamming Distance: ' + str(Hammanswer) + ',' +str(Hammanswer2)+ ',' +str(Hammanswer3))
print (np.flip(Hammperc[np.argsort(Hammperc)[-3:]]))
print('\n')
print('Minkowski Distance: '+str(Minkanswer)+ ',' +str(Minkanswer2)+ ',' +str(Minkanswer3))
print (np.flip(Minkperc[np.argsort(Minkperc)[-3:]]))
print('\n')



