from xml.dom import minidom
import re
import string
import numpy as np
import pandas as pd
import random
import math
from scipy.spatial import distance
from scipy.spatial.distance import cosine
##################################################################
stopword= open('Stopwords.txt','r').readlines()

stopwords_list = list(map(str.strip, stopword))
#print(stopwords_list)
##################################################################
MAX_ROWS = 1200
TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    
    return TAG_RE.sub('', text)

#################### parse  xml file by name ###########################
printer = minidom.parse('Law.xml')
anime = minidom.parse('Anime.xml')
coffee = minidom.parse('Coffee.xml')


item1 = printer.getElementsByTagName('row')
item2 = anime.getElementsByTagName('row')
item3 = coffee.getElementsByTagName('row')
items=item1+item2+item3



############################## WORDMAP ######################################

def getbody(items):
    t=""
    count=0
    for item in items:
        count+=1
        if count==MAX_ROWS:
            break
        st = remove_tags(item.attributes['Body'].value)
        t=t+st
    return t
getbody(items)

s1=getbody(item1)
s2=getbody(item2)
s3=getbody(item3)
#print(s1)
#print(s2)
#print(s3)
s=s1+s2+s3
#print(s)

wordmap_dict = {}

def preprocess(s):
    
    index=0
    
    s = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', s)
    s = re.sub('[^\u0000-\u007F]+',' ',s)
    s = re.sub('\s',' ',s)
    import string
    exclude = set(string.punctuation)
    string= ''.join(c for c in s if c not in exclude)
    words = string.split(' ')
    

    for word in words:
        if word.lower() in stopwords_list:
            pass
        elif len(word)<=1:
            pass
        
        else:
            if word not in wordmap_dict:
                wordmap_dict[word] = index
                index += 1
    
    
    return wordmap_dict
preprocess(s)
 
#print(wordmap_dict)
#print('Length of wordmap: ',len(wordmap_dict))


#######################################################################
                        ### TRAINING DOCUMENTS ###

d1=[]
for i in range (0,MAX_ROWS):
    string1 = remove_tags(item1[i].attributes['Body'].value)
    string1 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', string1)
    string1 = re.sub('[^\u0000-\u007F]+',' ',string1)
    string1 = re.sub('\s',' ',string1)
    import string
    exclude = set(string.punctuation)
    string1= ''.join(c for c in string1 if c not in exclude)
    word1 = string1.split(" ")
    d1.append(word1)
    
dL1=np.array(d1)
#print(dL1)

d2=[]
for i in range (0,MAX_ROWS):
    string2 = remove_tags(item2[i].attributes['Body'].value)
    string2 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', string2)
    string2 = re.sub('[^\u0000-\u007F]+',' ',string2)
    string2 = re.sub('\s',' ',string2)
    import string
    exclude = set(string.punctuation)
    string2= ''.join(c for c in string2 if c not in exclude)
    word2 = string2.split(" ")
    d2.append(word2)
    
dL2=np.array(d2)
#print(dL2)

d3=[]
for i in range (0,MAX_ROWS):
    string3 = remove_tags(item3[i].attributes['Body'].value)
    string3 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', string3)
    string3 = re.sub('[^\u0000-\u007F]+',' ',string3)
    string3 = re.sub('\s',' ',string3)
    import string
    exclude = set(string.punctuation)
    string3= ''.join(c for c in string3 if c not in exclude)
    word3 = string3.split(" ")
    d3.append(word3)
    
dL3=np.array(d3)
#print(dL3)

################################ HAMMING TRAINING VECTOR #######################

df1=[]
for i in range(0,MAX_ROWS):
    vector1 = [0]*len(wordmap_dict)
    for w1 in dL1[i]:
        if w1 in wordmap_dict.keys():
            vector1[wordmap_dict[w1]]=1
    c1=list(vector1)
    df1.append(c1)
       
dlv1=np.array(df1)   
#print('dlv1: ',dlv1)


df2=[]
for i in range(0,MAX_ROWS):
    vector2 = [0]*len(wordmap_dict)
    for w2 in dL2[i]:
        if w2 in wordmap_dict.keys():
            vector2[wordmap_dict[w2]]=1
    c2=list(vector2)
    df2.append(c2)
      
dlv2=np.array(df2)   
#print('dlv2: ',dlv2)


df3=[]
for i in range(0,MAX_ROWS):
    vector3 = [0]*len(wordmap_dict)
    for w3 in dL3[i]:
        if w3 in wordmap_dict.keys():
            vector3[wordmap_dict[w3]]=1
    c3=list(vector3)
    df3.append(c3)
   
dlv3=np.array(df3)   
#print('dlv3: ',dlv3)


############################# TEST ROWS ###################################

testdoc1 = minidom.parse('lawtest.xml')
testdoc2 = minidom.parse('animetest.xml')
testdoc3 = minidom.parse('coffeetest.xml')

testItems1 = testdoc1.getElementsByTagName('row')
testItems2 = testdoc2.getElementsByTagName('row')
testItems3 = testdoc3.getElementsByTagName('row')

MAX_ROWS_TEST=50
td1=[]
for i in range (0,MAX_ROWS_TEST):
    tstring1 = remove_tags(testItems1[i].attributes['Body'].value)
    tstring1 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', tstring1)
    tstring1 = re.sub('[^\u0000-\u007F]+',' ',tstring1)
    tstring1 = re.sub('\s',' ',tstring1)
    import string
    exclude = set(string.punctuation)
    tstring1= ''.join(c for c in tstring1 if c not in exclude)
    tword1 = tstring1.split(" ")
    td1.append(tword1)
    
tdL1=np.array(td1)
#print(tdL1)

td2=[]
for i in range (0,MAX_ROWS_TEST):
    tstring2 = remove_tags(testItems2[i].attributes['Body'].value)
    tstring2 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', tstring2)
    tstring2 = re.sub('[^\u0000-\u007F]+',' ',tstring2)
    tstring2 = re.sub('\s',' ',tstring2)
    import string
    exclude = set(string.punctuation)
    tstring2= ''.join(c for c in tstring2 if c not in exclude)
    tword2 = tstring2.split(" ")
    td2.append(tword2)
    
tdL2=np.array(td2)
#print(tdL2)

td3=[]
for i in range (0,MAX_ROWS_TEST):
    tstring3 = remove_tags(testItems3[i].attributes['Body'].value)
    tstring3 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', tstring3)
    tstring3 = re.sub('[^\u0000-\u007F]+',' ',tstring3)
    tstring3 = re.sub('\s',' ',tstring3)
    import string
    exclude = set(string.punctuation)
    tstring3= ''.join(c for c in tstring3 if c not in exclude)
    tword3 = tstring3.split(" ")
    td3.append(tword3)
    
tdL3=np.array(td3)
#print(tdL3) 


###################### TEST VECTOR FOR HAMMING ############################################################

tdf1=[]
for i in range(0,MAX_ROWS_TEST):
    testvector1 = [0]*len(wordmap_dict)
    for tw1 in tdL1[i]:
        if tw1 in wordmap_dict.keys():
            testvector1[wordmap_dict[tw1]]=1
    tc1=list(testvector1)
    tdf1.append(tc1)
       
tdlv1=np.array(tdf1)   
#print('tdlv1: ',tdlv1)


tdf2=[]
for i in range(0,MAX_ROWS_TEST):
    testvector2 = [0]*len(wordmap_dict)
    for tw2 in tdL2[i]:
        if tw2 in wordmap_dict.keys():
            testvector2[wordmap_dict[tw2]]=1
    tc2=list(testvector2)
    tdf2.append(tc2)
      
tdlv2=np.array(tdf2)   
#print('tdlv2: ',tdlv2)


tdf3=[]
for i in range(0,MAX_ROWS_TEST):
    testvector3 = [0]*len(wordmap_dict)
    for tw3 in tdL3[i]:
        if tw3 in wordmap_dict.keys():
            testvector3[wordmap_dict[tw3]]=1
    tc3=list(testvector3)
    tdf3.append(tc3)
   
tdlv3=np.array(tdf3)   
#print('tdlv3: ',tdlv3)

################################# HAMMING DISTANCE ############################################

############### K=1

#### For 3D_printer
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    #print('Hamming Distance: ',HD1)
    #print('Hamming Distance: ',HD2)
    #print('Hamming Distance: ',HD3)
    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    #print(hd1)
    #sorted_hd=np.sort(hd1)
    #print('sorted_hd: ',sorted_hd)
    #K=sorted_hd[:5]
    #print('K: ',K)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[0]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    if K.item(0)< int(1/3*len(hd1)):
        doc1+=1
    elif  K.item(0) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
        doc2+=1
    elif K.item(0)>=int(2/3*len(hd1)):
        doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1==1:
        print(x1)
        true_predictions +=1
    elif doc2==1:
        print(x2)
    elif doc3==1:
        print(x3)
    return true_predictions
p1=hamming_prediction(tdlv1[0])
p2=hamming_prediction(tdlv1[1])
p3=hamming_prediction(tdlv1[2])
p4=hamming_prediction(tdlv1[3])
p5=hamming_prediction(tdlv1[4])
p6=hamming_prediction(tdlv1[5])
p7=hamming_prediction(tdlv1[6])
p8=hamming_prediction(tdlv1[7])
p9=hamming_prediction(tdlv1[8])
p10=hamming_prediction(tdlv1[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy of LAW test file for K=1 is : ',Hacc,'%')

###### For Anime
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    #print('Hamming Distance: ',HD1)
    #print('Hamming Distance: ',HD2)
    #print('Hamming Distance: ',HD3)
    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    #print(hd1)
    #sorted_hd=np.sort(hd1)
    #print('sorted_hd: ',sorted_hd)
    #K=sorted_hd[:5]
    #print('K: ',K)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[0]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    if K.item(0)< int(1/3*len(hd1)):
        doc1+=1
    elif  K.item(0) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
        doc2+=1
    elif K.item(0)>=int(2/3*len(hd1)):
        doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1==1:
        print(x1)
        
    elif doc2==1:
        print(x2)
        true_predictions +=1
    elif doc3==1:
        print(x3)
    return true_predictions
p1=hamming_prediction(tdlv2[0])
p2=hamming_prediction(tdlv2[1])
p3=hamming_prediction(tdlv2[2])
p4=hamming_prediction(tdlv2[3])
p5=hamming_prediction(tdlv2[4])
p6=hamming_prediction(tdlv2[5])
p7=hamming_prediction(tdlv2[6])
p8=hamming_prediction(tdlv2[7])
p9=hamming_prediction(tdlv2[8])
p10=hamming_prediction(tdlv2[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy of Anime test file for K=1 is : ',Hacc,'%')

### For Coffee
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    #print('Hamming Distance: ',HD1)
    #print('Hamming Distance: ',HD2)
    #print('Hamming Distance: ',HD3)
    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    #print(hd1)
    #sorted_hd=np.sort(hd1)
    #print('sorted_hd: ',sorted_hd)
    #K=sorted_hd[:5]
    #print('K: ',K)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[0]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    if K.item(0)< int(1/3*len(hd1)):
        doc1+=1
    elif  K.item(0) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
        doc2+=1
    elif K.item(0)>=int(2/3*len(hd1)):
        doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1==1:
        print(x1)
        
    elif doc2==1:
        print(x2)
    elif doc3==1:
        print(x3)
        true_predictions +=1
    return true_predictions
p1=hamming_prediction(tdlv3[0])
p2=hamming_prediction(tdlv3[1])
p3=hamming_prediction(tdlv3[2])
p4=hamming_prediction(tdlv3[3])
p5=hamming_prediction(tdlv3[4])
p6=hamming_prediction(tdlv3[5])
p7=hamming_prediction(tdlv3[6])
p8=hamming_prediction(tdlv3[7])
p9=hamming_prediction(tdlv3[8])
p10=hamming_prediction(tdlv3[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy of Coffee test file for K=1 is : ',Hacc,'%')


###################################################  K=3
#### For 3D_printer
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[:3]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    for j in range(0,3):
            if K.item(j)< int(1/3*len(hd1)):
                doc1+=1
            elif  K.item(j) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
                doc2+=1
            elif K.item(j)>=int(2/3*len(hd1)):
                doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1>=2:
        print(x1)
        true_predictions +=1
    elif doc2>=2:
        print(x2)
    elif doc3>=2:
        print(x3)
    elif doc1==1 and doc2==1 and doc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x1:
            true_predictions+=1
    return true_predictions
    
p1=hamming_prediction(tdlv1[0])
p2=hamming_prediction(tdlv1[1])
p3=hamming_prediction(tdlv1[2])
p4=hamming_prediction(tdlv1[3])
p5=hamming_prediction(tdlv1[4])
p6=hamming_prediction(tdlv1[5])
p7=hamming_prediction(tdlv1[6])
p8=hamming_prediction(tdlv1[7])
p9=hamming_prediction(tdlv1[8])
p10=hamming_prediction(tdlv1[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy of LAW test file for K=3 is : ',Hacc,'%')

### For Anime
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[:3]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    for j in range(0,3):
            if K.item(j)< int(1/3*len(hd1)):
                doc1+=1
            elif  K.item(j) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
                doc2+=1
            elif K.item(j)>=int(2/3*len(hd1)):
                doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1>=2:
        print(x1)
    elif doc2>=2:
        print(x2)
        true_predictions +=1
    elif doc3>=2:
        print(x3)
    elif doc1==1 and doc2==1 and doc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x2:
            true_predictions+=1
    return true_predictions
    
p1=hamming_prediction(tdlv2[0])
p2=hamming_prediction(tdlv2[1])
p3=hamming_prediction(tdlv2[2])
p4=hamming_prediction(tdlv2[3])
p5=hamming_prediction(tdlv2[4])
p6=hamming_prediction(tdlv2[5])
p7=hamming_prediction(tdlv2[6])
p8=hamming_prediction(tdlv2[7])
p9=hamming_prediction(tdlv2[8])
p10=hamming_prediction(tdlv2[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy Of Anime test file for K=3 is : ',Hacc,'%')

### For Coffee
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[:3]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    for j in range(0,3):
            if K.item(j)< int(1/3*len(hd1)):
                doc1+=1
            elif  K.item(j) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
                doc2+=1
            elif K.item(j)>=int(2/3*len(hd1)):
                doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1>=2:
        print(x1)
    elif doc2>=2:
        print(x2)
    elif doc3>=2:
        print(x3)
        true_predictions +=1
    elif doc1==1 and doc2==1 and doc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    return true_predictions
    
p1=hamming_prediction(tdlv2[0])
p2=hamming_prediction(tdlv2[1])
p3=hamming_prediction(tdlv2[2])
p4=hamming_prediction(tdlv2[3])
p5=hamming_prediction(tdlv2[4])
p6=hamming_prediction(tdlv2[5])
p7=hamming_prediction(tdlv2[6])
p8=hamming_prediction(tdlv2[7])
p9=hamming_prediction(tdlv2[8])
p10=hamming_prediction(tdlv2[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy Of Coffee test file for K=3 is : ',Hacc,'%')


##################################################### K=5
### For 3D_printer
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[:5]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    for j in range(0,5):
            if K.item(j)< int(1/3*len(hd1)):
                doc1+=1
            elif  K.item(j) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
                doc2+=1
            elif K.item(j)>=int(2/3*len(hd1)):
                doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1>=3:
        print(x1)
        true_predictions+=1
    elif doc2>=3:
        print(x2)
    elif doc3>=3:
        print(x3)
    elif doc1==2 and doc2==2:
        z= random.choice([x1,x2])
        print(z)
        if z== x1:
            true_predictions+=1
    elif doc1==2 and doc3==2:
        z= random.choice([x1,x3])
        print(z)
        if z== x1:
            true_predictions+=1
    elif doc2==2 and doc3==2:
        z= random.choice([x2,x3])
        print(z)
    return true_predictions
p1=hamming_prediction(tdlv1[0])
p2=hamming_prediction(tdlv1[1])
p3=hamming_prediction(tdlv1[2])
p4=hamming_prediction(tdlv1[3])
p5=hamming_prediction(tdlv1[4])
p6=hamming_prediction(tdlv1[5])
p7=hamming_prediction(tdlv1[6])
p8=hamming_prediction(tdlv1[7])
p9=hamming_prediction(tdlv1[8])
p10=hamming_prediction(tdlv1[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy of LAW test file for K=5 is : ',Hacc,'%')

### For Anime
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[:5]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    for j in range(0,5):
            if K.item(j)< int(1/3*len(hd1)):
                doc1+=1
            elif  K.item(j) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
                doc2+=1
            elif K.item(j)>=int(2/3*len(hd1)):
                doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1>=3:
        print(x1)
    elif doc2>=3:
        print(x2)
        true_predictions+=1
    elif doc3>=3:
        print(x3)
    elif doc1==2 and doc2==2:
        z= random.choice([x1,x2])
        print(z)
        if z== x2:
            true_predictions+=1
    elif doc1==2 and doc3==2:
        z= random.choice([x1,x3])
        print(z)
    elif doc2==2 and doc3==2:
        z= random.choice([x2,x3])
        print(z)
        if z== x2:
            true_predictions+=1
    return true_predictions
p1=hamming_prediction(tdlv2[0])
p2=hamming_prediction(tdlv2[1])
p3=hamming_prediction(tdlv2[2])
p4=hamming_prediction(tdlv2[3])
p5=hamming_prediction(tdlv2[4])
p6=hamming_prediction(tdlv2[5])
p7=hamming_prediction(tdlv2[6])
p8=hamming_prediction(tdlv2[7])
p9=hamming_prediction(tdlv2[8])
p10=hamming_prediction(tdlv2[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy of Anime test file for K=5 is : ',Hacc,'%')

### For Coffee
def hamming_prediction(vv):
    
    H1=[]
    for m in range (0, len(dlv1)):
        h1=np.count_nonzero(dlv1[m]!=vv)
        H1.append(h1)
    HD1=np.array(H1)

    H2=[]
    for m in range (0, len(dlv2)):
        h2=np.count_nonzero(dlv2[m]!=vv)
        H2.append(h2)
    HD2=np.array(H2)

    H3=[]
    for m in range (0, len(dlv3)):
        h3=np.count_nonzero(dlv3[m]!=vv)
        H3.append(h3)
    HD3=np.array(H3)

    hd=np.append(HD1,HD2)
    hd1=np.append(hd,HD3)
    loc=np.argsort(hd1)
    #print(loc)
    K=loc[:5]
    #print(K)

    doc1=0
    doc2=0
    doc3=0
    for j in range(0,5):
            if K.item(j)< int(1/3*len(hd1)):
                doc1+=1
            elif  K.item(j) in range(int (1/3*len(hd1)),int (2/3*len(hd1))):
                doc2+=1
            elif K.item(j)>=int(2/3*len(hd1)):
                doc3+=1

    #print(doc1)
    #print(doc2)
    #print(doc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if doc1>=3:
        print(x1)
    elif doc2>=3:
        print(x2)
    elif doc3>=3:
        print(x3)
        true_predictions+=1
    elif doc1==2 and doc2==2:
        z= random.choice([x1,x2])
        print(z)
    elif doc1==2 and doc3==2:
        z= random.choice([x1,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    elif doc2==2 and doc3==2:
        z= random.choice([x2,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    return true_predictions
p1=hamming_prediction(tdlv3[0])
p2=hamming_prediction(tdlv3[1])
p3=hamming_prediction(tdlv3[2])
p4=hamming_prediction(tdlv3[3])
p5=hamming_prediction(tdlv3[4])
p6=hamming_prediction(tdlv3[5])
p7=hamming_prediction(tdlv3[6])
p8=hamming_prediction(tdlv3[7])
p9=hamming_prediction(tdlv3[8])
p10=hamming_prediction(tdlv3[9])

tru=p1+p2+p3+p4+p5+p6+p7+p8+p9+p10
total_predictions=10
Hacc=(tru/total_predictions)*100
print('Hamming Distance Accuracy of Coffee test file for K=5 is : ',Hacc,'%')

####################################################################
     ### EUCLIDEAN & COSINE SIMILARITY TRAINING VECTOR ###

ef1=[]
for i in range(0,MAX_ROWS):
    vect1 = [0]*len(wordmap_dict)
    for y1 in dL1[i]:
        if y1 in wordmap_dict.keys():
            vect1[wordmap_dict[y1]]+=1
    p1=list(vect1)
    ef1.append(p1)
       
elv1=np.array(ef1)   
#print('elv1: ',elv1)


ef2=[]
for i in range(0,MAX_ROWS):
    vect2 = [0]*len(wordmap_dict)
    for y2 in dL2[i]:
        if y2 in wordmap_dict.keys():
            vect2[wordmap_dict[y2]]+=1
    p2=list(vect2)
    ef2.append(p2)
      
elv2=np.array(ef2)   
#print('elv2: ',elv2)


ef3=[]
for i in range(0,MAX_ROWS):
    vect3 = [0]*len(wordmap_dict)
    for y3 in dL3[i]:
        if y3 in wordmap_dict.keys():
            vect3[wordmap_dict[y3]]+=1
    p3=list(vect3)
    ef3.append(p3)
   
elv3=np.array(ef3)   
#print('elv3: ',elv3)


############################# EUCLIDEAN & COSINE TEST VECTOR ###################################
tde1=[]
for i in range(0,MAX_ROWS_TEST):
    testvect1 = [0]*len(wordmap_dict)
    for wt1 in tdL1[i]:
        if wt1 in wordmap_dict.keys():
            testvect1[wordmap_dict[wt1]]+=1
    te1=list(testvect1)
    tde1.append(te1)
   
tdle1=np.array(tde1)   
#print('tdle1: ',tdle1)


tde2=[]
for i in range(0,MAX_ROWS_TEST):
    testvect2 = [0]*len(wordmap_dict)
    for wt2 in tdL2[i]:
        if wt2 in wordmap_dict.keys():
            testvect2[wordmap_dict[wt2]]+=1
    te2=list(testvect2)
    tde2.append(te2)
   
tdle2=np.array(tde2)   
#print('tdle2: ',tdle2)


tde3=[]
for i in range(0,MAX_ROWS_TEST):
    testvect3 = [0]*len(wordmap_dict)
    for wt3 in tdL3[i]:
        if wt3 in wordmap_dict.keys():
            testvect3[wordmap_dict[wt3]]+=1
    te3=list(testvect3)
    tde3.append(te3)
   
tdle3=np.array(tde3)   
#print('tdle3: ',tdle3)


###############################################################
                   ###KNN EUCLIDEAN PREDICTION###

############### K=1
### For 3D_printer

def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[0]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    
    if N.item(0)< int(1/3*len(ed1)):
        Edoc1+=1
    elif  N.item(0) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
        Edoc2+=1
    elif N.item(0)>= int(2/3*len(ed1)):
        Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1 ==1:
        print(x1)
        true_predictions+=1
    elif Edoc2 ==1:
        print(x2)
    elif Edoc3 ==1:
        print(x3)
    return true_predictions

E1=Euclidean_prediction(tdle1[0])
E2=Euclidean_prediction(tdle1[1])
E3=Euclidean_prediction(tdle1[2])
E4=Euclidean_prediction(tdle1[3])
E5=Euclidean_prediction(tdle1[4])
E6=Euclidean_prediction(tdle1[5])
E7=Euclidean_prediction(tdle1[6])
E8=Euclidean_prediction(tdle1[7])
E9=Euclidean_prediction(tdle1[8])
E10=Euclidean_prediction(tdle1[9])

tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of LAW test file for K=1 is : ',Eacc,'%')

### For Anime
def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[0]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    
    if N.item(0)< int(1/3*len(ed1)):
        Edoc1+=1
    elif  N.item(0) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
        Edoc2+=1
    elif N.item(0)>= int(2/3*len(ed1)):
        Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1 ==1:
        print(x1)
    elif Edoc2 ==1:
        print(x2)
        true_predictions+=1
    elif Edoc3 ==1:
        print(x3)
    return true_predictions

E1=Euclidean_prediction(tdle2[0])
E2=Euclidean_prediction(tdle2[1])
E3=Euclidean_prediction(tdle2[2])
E4=Euclidean_prediction(tdle2[3])
E5=Euclidean_prediction(tdle2[4])
E6=Euclidean_prediction(tdle2[5])
E7=Euclidean_prediction(tdle2[6])
E8=Euclidean_prediction(tdle2[7])
E9=Euclidean_prediction(tdle2[8])
E10=Euclidean_prediction(tdle2[9])

tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of Anime test file for K=1 is : ',Eacc,'%')

### For Coffee
def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[0]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    
    if N.item(0)< int(1/3*len(ed1)):
        Edoc1+=1
    elif  N.item(0) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
        Edoc2+=1
    elif N.item(0)>= int(2/3*len(ed1)):
        Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1 ==1:
        print(x1)
    elif Edoc2 ==1:
        print(x2)
    elif Edoc3 ==1:
        print(x3)
        true_predictions+=1
    return true_predictions

E1=Euclidean_prediction(tdle3[0])
E2=Euclidean_prediction(tdle3[1])
E3=Euclidean_prediction(tdle3[2])
E4=Euclidean_prediction(tdle3[3])
E5=Euclidean_prediction(tdle3[4])
E6=Euclidean_prediction(tdle3[5])
E7=Euclidean_prediction(tdle3[6])
E8=Euclidean_prediction(tdle3[7])
E9=Euclidean_prediction(tdle3[8])
E10=Euclidean_prediction(tdle3[9])

tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of Coffee test file for K=1 is : ',Eacc,'%')

##################### K=3
### For 3d_printer

def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[:3]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    for s in range(0,3):
            if N.item(s)< int(1/3*len(ed1)):
                Edoc1+=1
            elif  N.item(s) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
                Edoc2+=1
            elif N.item(s)>= int(2/3*len(ed1)):
                Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1 >=2:
        print(x1)
        true_predictions+=1
    elif Edoc2 >=2:
        print(x2)
    elif Edoc3 >=2:
        print(x3)
    elif Edoc1==1 and Edoc2==1 and Edoc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x1:
            true_predictions+=1
    return true_predictions

E1=Euclidean_prediction(tdle1[0])
E2=Euclidean_prediction(tdle1[1])
E3=Euclidean_prediction(tdle1[2])
E4=Euclidean_prediction(tdle1[3])
E5=Euclidean_prediction(tdle1[4])
E6=Euclidean_prediction(tdle1[5])
E7=Euclidean_prediction(tdle1[6])
E8=Euclidean_prediction(tdle1[7])
E9=Euclidean_prediction(tdle1[8])
E10=Euclidean_prediction(tdle1[9])

tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of LAW test file for K=3 is : ',Eacc,'%')

### For Anime
def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[:3]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    for s in range(0,3):
            if N.item(s)< int(1/3*len(ed1)):
                Edoc1+=1
            elif  N.item(s) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
                Edoc2+=1
            elif N.item(s)>= int(2/3*len(ed1)):
                Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1 >=2:
        print(x1)
    elif Edoc2 >=2:
        print(x2)
        true_predictions+=1
    elif Edoc3 >=2:
        print(x3)
    elif Edoc1==1 and Edoc2==1 and Edoc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x2:
            true_predictions+=1
    return true_predictions

E1=Euclidean_prediction(tdle2[0])
E2=Euclidean_prediction(tdle2[1])
E3=Euclidean_prediction(tdle2[2])
E4=Euclidean_prediction(tdle2[3])
E5=Euclidean_prediction(tdle2[4])
E6=Euclidean_prediction(tdle2[5])
E7=Euclidean_prediction(tdle2[6])
E8=Euclidean_prediction(tdle2[7])
E9=Euclidean_prediction(tdle2[8])
E10=Euclidean_prediction(tdle2[9])

tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of Anime test file for K=3 is : ',Eacc,'%')

### For Coffee
def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[:3]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    for s in range(0,3):
            if N.item(s)< int(1/3*len(ed1)):
                Edoc1+=1
            elif  N.item(s) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
                Edoc2+=1
            elif N.item(s)>= int(2/3*len(ed1)):
                Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1 >=2:
        print(x1)
    elif Edoc2 >=2:
        print(x2)
    elif Edoc3 >=2:
        print(x3)
        true_predictions+=1
    elif Edoc1==1 and Edoc2==1 and Edoc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    return true_predictions

E1=Euclidean_prediction(tdle3[0])
E2=Euclidean_prediction(tdle3[1])
E3=Euclidean_prediction(tdle3[2])
E4=Euclidean_prediction(tdle3[3])
E5=Euclidean_prediction(tdle3[4])
E6=Euclidean_prediction(tdle3[5])
E7=Euclidean_prediction(tdle3[6])
E8=Euclidean_prediction(tdle3[7])
E9=Euclidean_prediction(tdle3[8])
E10=Euclidean_prediction(tdle3[9])

tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of Coffee test file for K=3 is : ',Eacc,'%')

################## K=5

### For 3D_Printer
def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[:5]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    for s in range(0,5):
            if N.item(s)< int(1/3*len(ed1)):
                Edoc1+=1
            elif  N.item(s) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
                Edoc2+=1
            elif N.item(s)>= int(2/3*len(ed1)):
                Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1>=3:
        print(x1)
        true_predictions+=1
    elif Edoc2>=3:
        print(x2)
    elif Edoc3>=3:
        print(x3)
    elif Edoc1==2 and Edoc2==2:
        z= random.choice([x1,x2])
        print(z)
        if z== x1:
            true_predictions+=1
    elif Edoc1==2 and Edoc3==2:
        z= random.choice([x1,x3])
        print(z)
        if z== x1:
            true_predictions+=1
    elif Edoc2==2 and Edoc3==2:
        z= random.choice([x2,x3])
        print(z)
    return true_predictions

E1=Euclidean_prediction(tdle1[0])
E2=Euclidean_prediction(tdle1[1])
E3=Euclidean_prediction(tdle1[2])
E4=Euclidean_prediction(tdle1[3])
E5=Euclidean_prediction(tdle1[4])
E6=Euclidean_prediction(tdle1[5])
E7=Euclidean_prediction(tdle1[6])
E8=Euclidean_prediction(tdle1[7])
E9=Euclidean_prediction(tdle1[8])
E10=Euclidean_prediction(tdle1[9])

tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of LAW test file for K=5 is : ',Eacc,'%')

### For Anime
def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[:5]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    for s in range(0,5):
            if N.item(s)< int(1/3*len(ed1)):
                Edoc1+=1
            elif  N.item(s) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
                Edoc2+=1
            elif N.item(s)>= int(2/3*len(ed1)):
                Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1>=3:
        print(x1)
    elif Edoc2>=3:
        print(x2)
        true_predictions+=1
    elif Edoc3>=3:
        print(x3)
    elif Edoc1==2 and Edoc2==2:
        z= random.choice([x1,x2])
        print(z)
        if z== x2:
            true_predictions+=1
    elif Edoc1==2 and Edoc3==2:
        z= random.choice([x1,x3])
        print(z)
    elif Edoc2==2 and Edoc3==2:
        z= random.choice([x2,x3])
        print(z)
        if z== x2:
            true_predictions+=1
    return true_predictions

E1=Euclidean_prediction(tdle2[0])
E2=Euclidean_prediction(tdle2[1])
E3=Euclidean_prediction(tdle2[2])
E4=Euclidean_prediction(tdle2[3])
E5=Euclidean_prediction(tdle2[4])
E6=Euclidean_prediction(tdle2[5])
E7=Euclidean_prediction(tdle2[6])
E8=Euclidean_prediction(tdle2[7])
E9=Euclidean_prediction(tdle2[8])
E10=Euclidean_prediction(tdle2[9])


tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of Anime test file for K=5 is : ',Eacc,'%')

### For Coffee
def Euclidean_prediction(cc):
    D1=[]
    for k in range (0, len(elv1)):
        E1 = distance.euclidean(elv1[k],cc)
        D1.append(E1)
    ED1=np.array(D1)
    #print(ED1)

    D2=[]
    for k in range (0, len(elv2)):
        E2 = distance.euclidean(elv2[k],cc)
        D2.append(E2)
    ED2=np.array(D2)
    #print(ED2)

    D3=[]
    for k in range (0, len(elv3)):
        E3 = distance.euclidean(elv3[k],cc)
        D3.append(E3)
    ED3=np.array(D3)
    #print(ED3)

    ed=np.append(ED1,ED2)
    ed1=np.append(ed,ED3)
    locED=np.argsort(ed1)
    #print(locED)
    N=locED[:5]
    #print(N)

    Edoc1=0
    Edoc2=0
    Edoc3=0
    for s in range(0,5):
            if N.item(s)< int(1/3*len(ed1)):
                Edoc1+=1
            elif  N.item(s) in range(int (1/3*len(ed1)),int (2/3*len(ed1))):
                Edoc2+=1
            elif N.item(s)>= int(2/3*len(ed1)):
                Edoc3+=1

    #print(Edoc1)
    #print(Edoc2)
    #print(Edoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0

    if Edoc1>=3:
        print(x1)
    elif Edoc2>=3:
        print(x2)
    elif Edoc3>=3:
        print(x3)
        true_predictions+=1
    elif Edoc1==2 and Edoc2==2:
        z= random.choice([x1,x2])
        print(z)
    elif Edoc1==2 and Edoc3==2:
        z= random.choice([x1,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    elif Edoc2==2 and Edoc3==2:
        z= random.choice([x2,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    return true_predictions

E1=Euclidean_prediction(tdle3[0])
E2=Euclidean_prediction(tdle3[1])
E3=Euclidean_prediction(tdle3[2])
E4=Euclidean_prediction(tdle3[3])
E5=Euclidean_prediction(tdle3[4])
E6=Euclidean_prediction(tdle3[5])
E7=Euclidean_prediction(tdle3[6])
E8=Euclidean_prediction(tdle3[7])
E9=Euclidean_prediction(tdle3[8])
E10=Euclidean_prediction(tdle3[9])

tru=E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
total_predictions=10
Eacc=(tru/total_predictions)*100
print('Euclidean Distance Accuracy of Coffee test file for K=5 is : ',Eacc,'%')


                         

###########################################################################
                  ### COSINE SIMILARITY TRAINING TFIDF ###

#l=list(wordmap_dict)
#wm=np.array(l)
#print(wm)
cdf=[]
CDF=[]
cdf1=np.concatenate((elv1,elv2),axis=0)
cdf2=np.concatenate((cdf1,elv3),axis=0)
cdf=pd.DataFrame(cdf2)
CDF = np.array(cdf2)
#print(cdf)

I=[]
def idf(cdf):
    _, n_docs = cdf.shape
    for c in cdf:
        df = np.count_nonzero(cdf[c])
        #print(df)
        i=math.log(float(n_docs) /(1+ df))
        I.append(i)
    return I
idf(cdf)
IDF=np.array(I)
#print(IDF)


F=[]

def tfidf(CDF):
    for r in range (0, len(CDF)):
        v=np.sum(CDF[r])
        T=[0]* len(CDF[0])
        x=[]
        for e in range (0, len(CDF[r])):
            np.seterr(divide='ignore', invalid='ignore')
            t= (CDF[r][e])/(v)
            T = t*IDF[e]
            x.append(T)
        F.append(x)
    return F
tfidf(CDF)
TFIDF=np.array(F)
#print(TFIDF)


########################### COSINE SIMILARITY TEST TFIDF ###################

tCDF=[]
tcdf1=np.concatenate((tdle1,tdle2),axis=0)
tcdf2=np.concatenate((tcdf1,tdle3),axis=0)
tCDF = np.array(tcdf2)
tF=[]

def tfidf_test(tCDF):
    for r in range (0, len(tCDF)):
        tv=np.sum(tCDF[r])
        tT=[0]* len(tCDF[0])
        tx=[]
        for e in range (0, len(tCDF[r])):
            np.seterr(divide='ignore', invalid='ignore')
            tt = (tCDF[r][e])/(tv)
            tT = tt*IDF[e]
            tx.append(tT)
        tF.append(tx)
    return tF
tfidf_test(tCDF)
TFIDF_test=np.array(tF)
#print(TFIDF_test)


############################ COSINE SIMILARITY ###############################

################### K=1
### For 3D_Printer

CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-1]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    
    if csloc.item(-1)< int(1/3*len(cosine_s)):
        cdoc1+=1
    elif  csloc.item(-1) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
        cdoc2+=1
    elif csloc.item(-1)>= int(2/3*len(cosine_s)):
        cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1==1:
        print(x1)
        true_predictions+=1
    elif cdoc2==1:
        print(x2)
    elif cdoc3==1:
        print(x3)
    return true_predictions

C1=Cosine_prediction(TFIDF_test[0])
C2=Cosine_prediction(TFIDF_test[1])
C3=Cosine_prediction(TFIDF_test[2])
C4=Cosine_prediction(TFIDF_test[3])
C5=Cosine_prediction(TFIDF_test[4])
C6=Cosine_prediction(TFIDF_test[5])
C7=Cosine_prediction(TFIDF_test[6])
C8=Cosine_prediction(TFIDF_test[7])
C9=Cosine_prediction(TFIDF_test[8])
C10=Cosine_prediction(TFIDF_test[9])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy of LAW test file for K=1 is : ',Cacc,'%')

### For Anime

CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-1]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    
    if csloc.item(-1)< int(1/3*len(cosine_s)):
        cdoc1+=1
    elif  csloc.item(-1) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
        cdoc2+=1
    elif csloc.item(-1)>= int(2/3*len(cosine_s)):
        cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1==1:
        print(x1)
    elif cdoc2==1:
        print(x2)
        true_predictions+=1
    elif cdoc3==1:
        print(x3)
    return true_predictions

C1=Cosine_prediction(TFIDF_test[51])
C2=Cosine_prediction(TFIDF_test[52])
C3=Cosine_prediction(TFIDF_test[53])
C4=Cosine_prediction(TFIDF_test[54])
C5=Cosine_prediction(TFIDF_test[55])
C6=Cosine_prediction(TFIDF_test[56])
C7=Cosine_prediction(TFIDF_test[57])
C8=Cosine_prediction(TFIDF_test[58])
C9=Cosine_prediction(TFIDF_test[59])
C10=Cosine_prediction(TFIDF_test[60])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy of Anime test file for K=1 is : ',Cacc,'%')

### For Coffee

CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-1]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    
    if csloc.item(-1)< int(1/3*len(cosine_s)):
        cdoc1+=1
    elif  csloc.item(-1) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
        cdoc2+=1
    elif csloc.item(-1)>= int(2/3*len(cosine_s)):
        cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1==1:
        print(x1)
    elif cdoc2==1:
        print(x2)
    elif cdoc3==1:
        print(x3)
        true_predictions+=1
    return true_predictions

C1=Cosine_prediction(TFIDF_test[101])
C2=Cosine_prediction(TFIDF_test[102])
C3=Cosine_prediction(TFIDF_test[103])
C4=Cosine_prediction(TFIDF_test[104])
C5=Cosine_prediction(TFIDF_test[105])
C6=Cosine_prediction(TFIDF_test[106])
C7=Cosine_prediction(TFIDF_test[107])
C8=Cosine_prediction(TFIDF_test[108])
C9=Cosine_prediction(TFIDF_test[109])
C10=Cosine_prediction(TFIDF_test[110])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy of Coffee test file for K=1 is : ',Cacc,'%')

#################### K=3
### For 3D_Printer
CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-3:]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    for c in range(0,3):
        if csloc.item(c)< int(1/3*len(cosine_s)):
            cdoc1+=1
        elif  csloc.item(c) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
            cdoc2+=1
        elif csloc.item(c)>= int(2/3*len(cosine_s)):
            cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1>=2:
        print(x1)
        true_predictions+=1
    elif cdoc2>=2:
        print(x2)
    elif cdoc3>=2:
        print(x3)
    elif cdoc1==1 and cdoc2==1 and cdoc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x1:
            true_predictions+=1
    return true_predictions

C1=Cosine_prediction(TFIDF_test[0])
C2=Cosine_prediction(TFIDF_test[1])
C3=Cosine_prediction(TFIDF_test[2])
C4=Cosine_prediction(TFIDF_test[3])
C5=Cosine_prediction(TFIDF_test[4])
C6=Cosine_prediction(TFIDF_test[5])
C7=Cosine_prediction(TFIDF_test[6])
C8=Cosine_prediction(TFIDF_test[7])
C9=Cosine_prediction(TFIDF_test[8])
C10=Cosine_prediction(TFIDF_test[9])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy for of LAW test file K=3 is : ',Cacc,'%')

### For Anime
CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-3:]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    for c in range(0,3):
        if csloc.item(c)< int(1/3*len(cosine_s)):
            cdoc1+=1
        elif  csloc.item(c) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
            cdoc2+=1
        elif csloc.item(c)>= int(2/3*len(cosine_s)):
            cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1>=2:
        print(x1)
    elif cdoc2>=2:
        print(x2)
        true_predictions+=1
    elif cdoc3>=2:
        print(x3)
    elif cdoc1==1 and cdoc2==1 and cdoc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x2:
            true_predictions+=1
    return true_predictions

C1=Cosine_prediction(TFIDF_test[51])
C2=Cosine_prediction(TFIDF_test[52])
C3=Cosine_prediction(TFIDF_test[53])
C4=Cosine_prediction(TFIDF_test[54])
C5=Cosine_prediction(TFIDF_test[55])
C6=Cosine_prediction(TFIDF_test[56])
C7=Cosine_prediction(TFIDF_test[57])
C8=Cosine_prediction(TFIDF_test[58])
C9=Cosine_prediction(TFIDF_test[59])
C10=Cosine_prediction(TFIDF_test[60])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy for of Anime test file K=3 is : ',Cacc,'%')

### For Coffee
CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-3:]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    for c in range(0,3):
        if csloc.item(c)< int(1/3*len(cosine_s)):
            cdoc1+=1
        elif  csloc.item(c) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
            cdoc2+=1
        elif csloc.item(c)>= int(2/3*len(cosine_s)):
            cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1>=2:
        print(x1)
    elif cdoc2>=2:
        print(x2)
    elif cdoc3>=2:
        print(x3)
        true_predictions+=1
    elif cdoc1==1 and cdoc2==1 and cdoc3==1:
        z= random.choice([x1,x2,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    return true_predictions

C1=Cosine_prediction(TFIDF_test[101])
C2=Cosine_prediction(TFIDF_test[102])
C3=Cosine_prediction(TFIDF_test[103])
C4=Cosine_prediction(TFIDF_test[104])
C5=Cosine_prediction(TFIDF_test[105])
C6=Cosine_prediction(TFIDF_test[106])
C7=Cosine_prediction(TFIDF_test[107])
C8=Cosine_prediction(TFIDF_test[108])
C9=Cosine_prediction(TFIDF_test[109])
C10=Cosine_prediction(TFIDF_test[110])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy for of Coffee test file K=3 is : ',Cacc,'%')

################### K=5
### For 3D_Printer

CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-5:]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    for c in range(0,5):
        if csloc.item(c)< int(1/3*len(cosine_s)):
            cdoc1+=1
        elif  csloc.item(c) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
            cdoc2+=1
        elif csloc.item(c)>= int(2/3*len(cosine_s)):
            cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1>=3:
        print(x1)
        true_predictions+=1
    elif cdoc2>=3:
        print(x2)
    elif cdoc3>=3:
        print(x3)
    elif cdoc1==2 and cdoc2==2:
        z= random.choice([x1,x2])
        print(z)
        if z== x1:
            true_predictions+=1
    elif cdoc1==2 and cdoc3==2:
        z= random.choice([x1,x3])
        print(z)
        if z== x1:
            true_predictions+=1
    elif cdoc2==2 and cdoc3==2:
        z= random.choice([x2,x3])
        print(z)
    return true_predictions

C1=Cosine_prediction(TFIDF_test[0])
C2=Cosine_prediction(TFIDF_test[1])
C3=Cosine_prediction(TFIDF_test[2])
C4=Cosine_prediction(TFIDF_test[3])
C5=Cosine_prediction(TFIDF_test[4])
C6=Cosine_prediction(TFIDF_test[5])
C7=Cosine_prediction(TFIDF_test[6])
C8=Cosine_prediction(TFIDF_test[7])
C9=Cosine_prediction(TFIDF_test[8])
C10=Cosine_prediction(TFIDF_test[9])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy of LAW test file for K=5 is : ',Cacc,'%')

### For Anime

CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-5:]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    for c in range(0,5):
        if csloc.item(c)< int(1/3*len(cosine_s)):
            cdoc1+=1
        elif  csloc.item(c) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
            cdoc2+=1
        elif csloc.item(c)>= int(2/3*len(cosine_s)):
            cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1>=3:
        print(x1)
    elif cdoc2>=3:
        print(x2)
        true_predictions+=1
    elif cdoc3>=3:
        print(x3)
    elif cdoc1==2 and cdoc2==2:
        z= random.choice([x1,x2])
        print(z)
        if z== x2:
            true_predictions+=1
    elif cdoc1==2 and cdoc3==2:
        z= random.choice([x1,x3])
        print(z)
    elif cdoc2==2 and cdoc3==2:
        z= random.choice([x2,x3])
        print(z)
        if z== x2:
            true_predictions+=1
    return true_predictions

C1=Cosine_prediction(TFIDF_test[51])
C2=Cosine_prediction(TFIDF_test[52])
C3=Cosine_prediction(TFIDF_test[53])
C4=Cosine_prediction(TFIDF_test[54])
C5=Cosine_prediction(TFIDF_test[55])
C6=Cosine_prediction(TFIDF_test[56])
C7=Cosine_prediction(TFIDF_test[57])
C8=Cosine_prediction(TFIDF_test[58])
C9=Cosine_prediction(TFIDF_test[59])
C10=Cosine_prediction(TFIDF_test[60])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy of Anime test file for K=5 is : ',Cacc,'%')

### For Coffee

CS=[]
def Cosine_prediction(TFIDF_test):
    
    for i in range (0, len(TFIDF)):
        cs=(np.dot(TFIDF[i], TFIDF_test)) / (np.sqrt(np.sum(TFIDF[i]**2)) * np.sqrt(np.sum(TFIDF_test**2)))
        CS.append(cs)

    cosine_s=np.array(CS)
    #print(cosine_s)
    cosine_s[np.isnan(cosine_s)]=0
    #print(cosine_s)
    css= np.argsort(cosine_s)
    #print(css)
    csloc= css[-5:]
    #print(csloc)

    cdoc1=0
    cdoc2=0
    cdoc3=0
    for c in range(0,5):
        if csloc.item(c)< int(1/3*len(cosine_s)):
            cdoc1+=1
        elif  csloc.item(c) in range(int (1/3*len(cosine_s)),int (2/3*len(cosine_s))):
            cdoc2+=1
        elif csloc.item(c)>= int(2/3*len(cosine_s)):
            cdoc3+=1
    #print(cdoc1)
    #print(cdoc2)
    #print(cdoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if cdoc1>=3:
        print(x1)
    elif cdoc2>=3:
        print(x2)
    elif cdoc3>=3:
        print(x3)
        true_predictions+=1
    elif cdoc1==2 and cdoc2==2:
        z= random.choice([x1,x2])
        print(z)
        
    elif cdoc1==2 and cdoc3==2:
        z= random.choice([x1,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    elif cdoc2==2 and cdoc3==2:
        z= random.choice([x2,x3])
        print(z)
        if z== x3:
            true_predictions+=1
    return true_predictions

C1=Cosine_prediction(TFIDF_test[101])
C2=Cosine_prediction(TFIDF_test[102])
C3=Cosine_prediction(TFIDF_test[103])
C4=Cosine_prediction(TFIDF_test[104])
C5=Cosine_prediction(TFIDF_test[105])
C6=Cosine_prediction(TFIDF_test[106])
C7=Cosine_prediction(TFIDF_test[107])
C8=Cosine_prediction(TFIDF_test[108])
C9=Cosine_prediction(TFIDF_test[109])
C10=Cosine_prediction(TFIDF_test[110])

tru=C1+C2+C3+C4+C5+C6+C7+C8+C9+C10
total_predictions=10
Cacc=(tru/total_predictions)*100
print('Cosine Similarity Accuracy of Coffee test file for K=5 is : ',Cacc,'%')


###########################################################################################
                        ##### NAIVE BAYES TRAINING VECTOR ######

def pprocess(s):

    s = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', s)
    s = re.sub('[^\u0000-\u007F]+',' ',s)
    s = re.sub('\s',' ',s)
    import string
    exclude = set(string.punctuation)
    string= ''.join(c for c in s if c not in exclude)
    words = string.split(' ')
    return words
nx1=pprocess(s1)
nx2=pprocess(s2)
nx3=pprocess(s3)

nL1=np.array(nx1)
nL2=np.array(nx2)
nL3=np.array(nx3)
#print(nL1)
#print(nL2)
#print(nL3)

nf1=[]
nvect1 = [0]*len(wordmap_dict)
for v1 in nL1:
    if v1 in wordmap_dict.keys():
        nvect1[wordmap_dict[v1]]+=1
nf1.append(nvect1)      
nlv1=np.array(nf1)
t1=nlv1[0]
#print('t1: ',t1)

nf2=[]
nvect2 = [0]*len(wordmap_dict)
for v2 in nL2:
    if v2 in wordmap_dict.keys():
        nvect2[wordmap_dict[v2]]+=1
nf2.append(nvect2)      
nlv2=np.array(nf2)
t2=nlv2[0]
#print('t2: ',t2)

nf3=[]
nvect3 = [0]*len(wordmap_dict)
for v3 in nL3:
    if v3 in wordmap_dict.keys():
        nvect3[wordmap_dict[v3]]+=1
nf3.append(nvect3)      
nlv3=np.array(nf3)
t3=nlv3[0]
#print('t3: ',t3)

################################### NAIVE BAYES PREDICTION #######################################
### For 3D_Printer
def naivebayes(alpha):
    b=tdle1[0]
    m=len(wordmap_dict)
    d=len(nL1)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t1[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc1=np.prod(nd1)
    #print(Nc1)

    d=len(nL2)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t2[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc2=np.prod(nd1)
    #print(Nc2)

    d=len(nL3)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t3[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc3=np.prod(nd1)
    #print(Nc3)
    z1=np.append(Nc1,Nc2)
    z2=np.append(z1,Nc3)
    #print(z2)
    nbloc= np.argsort(z2)
    #print(nbloc)
    nbloc= nbloc[-1]
    #print(nbloc)

    ndoc1=0
    ndoc2=0
    ndoc3=0
    if nbloc.item(-1)==0:
        ndoc1+=1
    elif  nbloc.item(-1) ==1:
        ndoc2+=1
    elif nbloc.item(-1) ==2:
        ndoc3+=1
    #print(ndoc1)
    #print(ndoc2)
    #print(ndoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if ndoc1==1:
        print(x1)
        true_predictions+=1
    elif ndoc2==1:
        print(x2)
    elif ndoc3==1:
        print(x3)
    return true_predictions

NB1=naivebayes(alpha=.00001)
NB2=naivebayes(alpha=.0005)
NB3=naivebayes(alpha=.003)
NB4=naivebayes(alpha=.094)
NB5=naivebayes(alpha=.0073636)
NB6=naivebayes(alpha=.09999)
NB7=naivebayes(alpha=.10)
NB8=naivebayes(alpha=.9999)
NB9=naivebayes(alpha=.123)
NB10=naivebayes(alpha=12)
NB11=naivebayes(alpha=.0909349)
NB12=naivebayes(alpha=.00000364370)
NB13=naivebayes(alpha=.349384)
NB14=naivebayes(alpha=1.3493849837)
NB15=naivebayes(alpha=10.3403894398)
NB16=naivebayes(alpha=1.39483984)
NB17=naivebayes(alpha=1)
NB18=naivebayes(alpha=5)
NB18=naivebayes(alpha=7)
NB19=naivebayes(alpha=9)
NB20=naivebayes(alpha=10)
NB21=naivebayes(alpha=82)
NB22=naivebayes(alpha=77)
NB23=naivebayes(alpha=56)
NB24=naivebayes(alpha=122)
NB25=naivebayes(alpha=200)
NB26=naivebayes(alpha=45)
NB27=naivebayes(alpha=37)
NB28=naivebayes(alpha=16)
NB29=naivebayes(alpha=10)
NB30=naivebayes(alpha=20)
NB31=naivebayes(alpha=30)
NB32=naivebayes(alpha=40)
NB33=naivebayes(alpha=50)
NB34=naivebayes(alpha=60)
NB35=naivebayes(alpha=70)
NB36=naivebayes(alpha=80)
NB37=naivebayes(alpha=90)
NB38=naivebayes(alpha=910)
NB39=naivebayes(alpha=810)
NB40=naivebayes(alpha=710)
NB41=naivebayes(alpha=610)
NB42=naivebayes(alpha=510)
NB43=naivebayes(alpha=410)
NB44=naivebayes(alpha=210)
NB45=naivebayes(alpha=140)
NB46=naivebayes(alpha=310)
NB47=naivebayes(alpha=107)
NB48=naivebayes(alpha=1009)
NB49=naivebayes(alpha=509)
NB50=naivebayes(alpha=10000)

tru=(NB1+NB2+NB3+NB4+NB5+NB6+NB7+NB8+NB9+NB10+NB11+NB12+NB13+NB14+NB15+NB16+NB17+NB18
     +NB19+NB20+NB21+NB22+NB23+NB24+NB25+NB26+NB27+NB28+NB29+NB30+NB31+NB32+NB33+NB34
     +NB35+NB36+NB37+NB38+NB39+NB40+NB41+NB42+NB43+NB44+NB45+NB46+NB47+NB48+NB49+NB50)
     
     
     
total_predictions=50
NBacc=(tru/total_predictions)*100
print('Naive Bayes Accuracy of LAW test file taking 50 alpha values is : ',NBacc,'%')

### For Anime
def naivebayes(alpha):
    b=tdle2[0]
    m=len(wordmap_dict)
    d=len(nL1)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t1[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc1=np.prod(nd1)
    #print(Nc1)

    d=len(nL2)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t2[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc2=np.prod(nd1)
    #print(Nc2)

    d=len(nL3)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t3[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc3=np.prod(nd1)
    #print(Nc3)
    z1=np.append(Nc1,Nc2)
    z2=np.append(z1,Nc3)
    #print(z2)
    nbloc= np.argsort(z2)
    #print(nbloc)
    nbloc= nbloc[-1]
    #print(nbloc)

    ndoc1=0
    ndoc2=0
    ndoc3=0
    if nbloc.item(-1)==0:
        ndoc1+=1
    elif  nbloc.item(-1) ==1:
        ndoc2+=1
    elif nbloc.item(-1) ==2:
        ndoc3+=1
    #print(ndoc1)
    #print(ndoc2)
    #print(ndoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if ndoc1==1:
        print(x1)
    elif ndoc2==1:
        print(x2)
        true_predictions+=1
    elif ndoc3==1:
        print(x3)
    return true_predictions

NB1=naivebayes(alpha=.00001)
NB2=naivebayes(alpha=.0005)
NB3=naivebayes(alpha=.003)
NB4=naivebayes(alpha=.094)
NB5=naivebayes(alpha=.0073636)
NB6=naivebayes(alpha=.09999)
NB7=naivebayes(alpha=.10)
NB8=naivebayes(alpha=.9999)
NB9=naivebayes(alpha=.123)
NB10=naivebayes(alpha=12)
NB11=naivebayes(alpha=.0909349)
NB12=naivebayes(alpha=.00000364370)
NB13=naivebayes(alpha=.349384)
NB14=naivebayes(alpha=1.3493849837)
NB15=naivebayes(alpha=10.3403894398)
NB16=naivebayes(alpha=1.39483984)
NB17=naivebayes(alpha=1)
NB18=naivebayes(alpha=5)
NB18=naivebayes(alpha=7)
NB19=naivebayes(alpha=9)
NB20=naivebayes(alpha=10)
NB21=naivebayes(alpha=82)
NB22=naivebayes(alpha=77)
NB23=naivebayes(alpha=56)
NB24=naivebayes(alpha=122)
NB25=naivebayes(alpha=200)
NB26=naivebayes(alpha=45)
NB27=naivebayes(alpha=37)
NB28=naivebayes(alpha=16)
NB29=naivebayes(alpha=10)
NB30=naivebayes(alpha=20)
NB31=naivebayes(alpha=30)
NB32=naivebayes(alpha=40)
NB33=naivebayes(alpha=50)
NB34=naivebayes(alpha=60)
NB35=naivebayes(alpha=70)
NB36=naivebayes(alpha=80)
NB37=naivebayes(alpha=90)
NB38=naivebayes(alpha=910)
NB39=naivebayes(alpha=810)
NB40=naivebayes(alpha=710)
NB41=naivebayes(alpha=610)
NB42=naivebayes(alpha=510)
NB43=naivebayes(alpha=410)
NB44=naivebayes(alpha=210)
NB45=naivebayes(alpha=140)
NB46=naivebayes(alpha=310)
NB47=naivebayes(alpha=107)
NB48=naivebayes(alpha=1009)
NB49=naivebayes(alpha=509)
NB50=naivebayes(alpha=10000)

tru=(NB1+NB2+NB3+NB4+NB5+NB6+NB7+NB8+NB9+NB10+NB11+NB12+NB13+NB14+NB15+NB16+NB17+NB18
     +NB19+NB20+NB21+NB22+NB23+NB24+NB25+NB26+NB27+NB28+NB29+NB30+NB31+NB32+NB33+NB34
     +NB35+NB36+NB37+NB38+NB39+NB40+NB41+NB42+NB43+NB44+NB45+NB46+NB47+NB48+NB49+NB50)
     
     
     
total_predictions=50
NBacc=(tru/total_predictions)*100
print('Naive Bayes Accuracy of Anime test file taking 50 alpha values is : ',NBacc,'%')

### For Coffee
def naivebayes(alpha):
    b=tdle3[0]
    m=len(wordmap_dict)
    d=len(nL1)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t1[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc1=np.prod(nd1)
    #print(Nc1)

    d=len(nL2)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t2[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc2=np.prod(nd1)
    #print(Nc2)

    d=len(nL3)
    NA=[]
    for u in range (0, len(b)):
        nb=0
        if b[u] !=0 :
            a=t3[u]
            nb=(a+alpha)/(d+alpha*m)
            NA.append(nb)
    nd1=np.array(NA)
    #print(nd1)
    Nc3=np.prod(nd1)
    #print(Nc3)
    z1=np.append(Nc1,Nc2)
    z2=np.append(z1,Nc3)
    #print(z2)
    nbloc= np.argsort(z2)
    #print(nbloc)
    nbloc= nbloc[-1]
    #print(nbloc)

    ndoc1=0
    ndoc2=0
    ndoc3=0
    if nbloc.item(-1)==0:
        ndoc1+=1
    elif  nbloc.item(-1) ==1:
        ndoc2+=1
    elif nbloc.item(-1) ==2:
        ndoc3+=1
    #print(ndoc1)
    #print(ndoc2)
    #print(ndoc3)
    x1='LAW Document!!!'
    x2='ANIME Document!!!'
    x3='COFFEE Document!!!'
    true_predictions=0
    if ndoc1==1:
        print(x1)
    elif ndoc2==1:
        print(x2)
    elif ndoc3==1:
        print(x3)
        true_predictions+=1
    return true_predictions

NB1=naivebayes(alpha=.00001)
NB2=naivebayes(alpha=.0005)
NB3=naivebayes(alpha=.003)
NB4=naivebayes(alpha=.094)
NB5=naivebayes(alpha=.0073636)
NB6=naivebayes(alpha=.09999)
NB7=naivebayes(alpha=.10)
NB8=naivebayes(alpha=.9999)
NB9=naivebayes(alpha=.123)
NB10=naivebayes(alpha=12)
NB11=naivebayes(alpha=.0909349)
NB12=naivebayes(alpha=.00000364370)
NB13=naivebayes(alpha=.349384)
NB14=naivebayes(alpha=1.3493849837)
NB15=naivebayes(alpha=10.3403894398)
NB16=naivebayes(alpha=1.39483984)
NB17=naivebayes(alpha=1)
NB18=naivebayes(alpha=5)
NB18=naivebayes(alpha=7)
NB19=naivebayes(alpha=9)
NB20=naivebayes(alpha=10)
NB21=naivebayes(alpha=82)
NB22=naivebayes(alpha=77)
NB23=naivebayes(alpha=56)
NB24=naivebayes(alpha=122)
NB25=naivebayes(alpha=200)
NB26=naivebayes(alpha=45)
NB27=naivebayes(alpha=37)
NB28=naivebayes(alpha=16)
NB29=naivebayes(alpha=10)
NB30=naivebayes(alpha=20)
NB31=naivebayes(alpha=30)
NB32=naivebayes(alpha=40)
NB33=naivebayes(alpha=50)
NB34=naivebayes(alpha=60)
NB35=naivebayes(alpha=70)
NB36=naivebayes(alpha=80)
NB37=naivebayes(alpha=90)
NB38=naivebayes(alpha=910)
NB39=naivebayes(alpha=810)
NB40=naivebayes(alpha=710)
NB41=naivebayes(alpha=610)
NB42=naivebayes(alpha=510)
NB43=naivebayes(alpha=410)
NB44=naivebayes(alpha=210)
NB45=naivebayes(alpha=140)
NB46=naivebayes(alpha=310)
NB47=naivebayes(alpha=107)
NB48=naivebayes(alpha=1009)
NB49=naivebayes(alpha=509)
NB50=naivebayes(alpha=10000)

tru=(NB1+NB2+NB3+NB4+NB5+NB6+NB7+NB8+NB9+NB10+NB11+NB12+NB13+NB14+NB15+NB16+NB17+NB18
     +NB19+NB20+NB21+NB22+NB23+NB24+NB25+NB26+NB27+NB28+NB29+NB30+NB31+NB32+NB33+NB34
     +NB35+NB36+NB37+NB38+NB39+NB40+NB41+NB42+NB43+NB44+NB45+NB46+NB47+NB48+NB49+NB50)
     
     
     
total_predictions=50
NBacc=(tru/total_predictions)*100
print('Naive Bayes Accuracy of Coffee test file taking 50 alpha values is : ',NBacc,'%')




