import numpy as np
import pandas as pd
import math


data=pd.read_csv('train.csv', names=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','label'])
test_data=pd.read_csv('test.csv',names= ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','label'])
def entropy(label):
    entropy=0
    feat, counts= np.unique(label, return_counts=True)
    tot=np.sum(counts)
    for i in range(len(feat)):
        p=counts[i]/tot 
        if p!=0:
            entropy+= -p*math.log2(p)
    return entropy


def ME( label):
    attr, counts= np.unique(label, return_counts=True)
    tot=np.sum(counts)
    get_me=1-counts[np.argmax(counts)]/tot
    return get_me

def GI( label):
    feat,counts=np.unique(label, return_counts=True)
    tot=np.sum(counts)
    get_gi = 1
    for i in range(len(feat)):
        get_gi -=(counts[i]/tot)**2

    return get_gi


def information_entropy( file, attribute, label):
    
    weighted_entropy=0
    overall_entropy= entropy(file[label])
    feat, counts = np.unique(file[attribute], return_counts=True)
    tot=np.sum(counts)
    for i in range(len(feat)):
        weight_file=file[file[attribute]==feat[i] ]
        weighted_entropy+=counts[i]/tot*entropy(weight_file[label])

    inf_gain=overall_entropy- weighted_entropy
    return inf_gain

def me_g(file, attribute,label):
    weighted_me=0
    overall_me=ME(file[label])
    feat, counts =np.unique(file[attribute], return_counts=True)
    tot=np.sum(counts)
    for i in range(len(feat)):
        weight_file=file[file[attribute]==feat[i] ]
        weighted_me += counts[i]/tot*ME(weight_file[label])
    inf_gain=overall_me-weighted_me
    return inf_gain

def gi_g(file, attribute, label):
    weighted_gi=0
    overall_gi=GI(file[label])
    feat, counts=np.unique(file[attribute], return_counts=True)
    tot=np.sum(counts)
    for i in range(len(feat)):
        weight_file=file[file[attribute]== feat[i]]
        weighted_gi += counts[i]/tot * GI(weight_file[label])
    inf_gain=overall_gi - weighted_gi
    return inf_gain

def ID3_ent(depth, file, attribute, label ):
    feature, count= np.unique(file[label], return_counts=True)
    tot=np.sum(count)
    common=feature[np.argmax(count)]

   
    item_val=[information_entropy(file,i,label)  for i in attribute]
    best_feat_ind=np.argmax(item_val)
    best_feat=attribute[best_feat_ind]
    tree={best_feat:{}}
    for value in np.unique(file[best_feat]):
        sub_data = file[file[best_feat]== value]
        sub_feature, sub_count= np.unique(sub_data[label], return_counts=True)
        sub_common_label= sub_feature[np.argmax(sub_count)]

        if len(sub_data)==0 or depth==1:
            tree[best_feat][value] = sub_common_label

        else:
            subtree=ID3_ent(depth-1,sub_data,attribute,label)
            tree[best_feat][value]=subtree
        
    return tree

def ID3_me(depth,file, attribute, label ):
    
    feature, count= np.unique(file[label], return_counts=True)
    tot=np.sum(count)
    common=feature[np.argmax(count)]
    if len(np.unique(file[label])) <= 1:
        return np.unique(file[label])[0]
    elif len(attribute)==0:
        return common
    
    item_values=[ me_g(file,i,label) for i in attribute]
    best_feat_index = np.argmax(item_values)
    best_feat = attribute[best_feat_index]
        
    tree = {best_feat:{}}
    for value in np.unique(file[best_feat]):
        sub_data = file[file[best_feat]== value]
        sub_feature, sub_count= np.unique(sub_data[label], return_counts=True)
        sub_common_label= sub_feature[np.argmax(sub_count)]

        if len(sub_data)==0 or depth==1:
                tree[best_feat][value] = sub_common_label
        else:
            subtree=ID3_me(depth-1,sub_data,attribute,label)
            tree[best_feat][value]=subtree
    return tree

def ID3_gi(depth,file, attribute, label ):
    feature, count= np.unique(file[label], return_counts=True)
    #tot=np.sum(count)
    common=feature[np.argmax(count)]
    if len(np.unique(file[label])) <= 1:
        return np.unique(file[label])[0]
    elif len(attribute)==0:
        return common
    item_values=[ gi_g(file,i,label) for i in attribute]

    best_feat_index = np.argmax(item_values)
    best_feat = attribute[best_feat_index]
        
    tree = {best_feat:{}}

    for value in np.unique(file[best_feat]):
        sub_data = file[file[best_feat]== value]
        sub_feature, sub_count=np.unique(sub_data[label], return_counts=True)
        sub_common_label= sub_feature[np.argmax(sub_count)]

        if len(sub_data)==0 or depth==1:
            tree[best_feat][value] = sub_common_label
        else:
            subtree=ID3_gi(depth-1,sub_data,attribute,label)
            tree[best_feat][value]=subtree
    return tree

def predict(query,tree,default = 1):    
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result


def test(data,label,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
        
    return np.sum(predicted["predicted"] == label)/len(data)*100




def conv_num_to_bi(data,test_data, target_head="class"):
    med_d = data[target_head].median()
    min_d = data[target_head].min()
    max_d = data[target_head].max()
    bins = [min_d-1,med_d,max_d]
    data[target_head]=pd.cut(data[target_head],bins,labels=[0,1]) 
    test_data[target_head]=pd.cut(test_data[target_head],bins,labels=[0,1])


# In[9]:
# convert numeric data to binary
# train
Attributes= ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
Training_Label= data['label']
Training_Data= data[Attributes]

Test_Label= test_data['label']
Test_Data= test_data[Attributes]
conv_num_to_bi(data,test_data,'age')
conv_num_to_bi(data,test_data,'balance')
conv_num_to_bi(data,test_data,'day')
conv_num_to_bi(data,test_data,'duration')
conv_num_to_bi(data,test_data,'campaign')
conv_num_to_bi(data,test_data,'pdays')
conv_num_to_bi(data,test_data,'previous')

#--------------with unknown----------------------------------
def Common_value(data, label="class"):
    """ data : given dataset
        label: title of the label"""
    feature, f_count = np.unique(data[label],return_counts= True)
    max_count= np.argmax(f_count)
    if feature[max_count]=="unknown":
        f_count[max_count]=0
        common_label= feature[np.argmax(f_count)]
    else: 
        common_label = feature[max_count]
        
    
    return common_label
# Convert "unknown" values to the common values in the attributes.
com_Job= Common_value(data, label="job")
com_Education= Common_value(data, label="education")
com_Poutcome= Common_value(data, label="poutcome")
com_Contact= Common_value(data, label="contact")

data["poutcome"]=data.poutcome.replace("unknown", com_Poutcome)
data["job"]=data.job.replace("unknown", com_Job)
data["education"]=data.education.replace("unknown", com_Education)
data["contact"]=data.contact.replace("unknown", com_Contact)

test_data["poutcome"]=test_data.poutcome.replace("unknown", com_Poutcome)
test_data["job"]=test_data.job.replace("unknown", com_Job)
test_data["education"]=test_data.education.replace("unknown", com_Education)
test_data["contact"]=test_data.contact.replace("unknown", com_Contact)
#-------------------------------------------------------------
#-------------------------------------------------------------
print("With Information Gain,")
for i in range(16):
    tree=ID3_ent(i+1, data, Attributes,"label")
    train_acc1= test(data,Training_Label,tree)
    test_acc1 = test(test_data,Test_Label,tree)
    print("For depth ", i+1)
    print ("Accuracy in training is ", train_acc1)
    print("Accuracy in testing is ", test_acc1)


#-------------------------------------------------------------
print("Informatio gain using Majority Error,")

for i in range(16):
    tree=ID3_me(i+1, data, Attributes,"label")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree)
    print("For depth ", i+1)
    print ("Accuracy in training is", train_acc)
    print("Accuracy in testing is", test_acc)
    
#-------------------------------------------------------------
print("Information gain using Gini Index,")

for i in range(16):
    tree=ID3_gi(i+1, data, Attributes,"label")
    train_acc= test(data,Training_Label,tree)
    test_acc = test(test_data,Test_Label,tree) 
    print("For depth ", i+1)
    print ("Accuracy in training is", train_acc)
    print("Accuracy in testing is ", test_acc)