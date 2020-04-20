class Node:
    def __init__(self,right = None,left = None,split_feature_index = None,value = None,split_value = None):
        self.right = right
        self.left = left
        self.feature = split_feature_index
        self.value = value
        self.split_value = split_value
        
    def isPure(self):
        return self.value != None

class DecisionTreeClassifier:
    
    def __init__(self,max_depth = 100):
        self.max_depth = max_depth
        self.root = None

    
    
    def informationGain(self,feature,Y_train,split_pt):
        
        parent_entropy = entropyOfNode(Y_train)
        
        
        left_idx,right_idx = self.split(feature,split_pt)
        
        if len(left_idx) == 0 or len(right_idx) == 0:
            return -1
        
        n = len(Y_train)
        n_left = len(left_idx)
        n_right = len(right_idx)
        
        e_left = entropyOfNode(Y_train[left_idx])
        e_right = entropyOfNode(Y_train[right_idx])
        
        weighted_entropy = ((n_left/n)*e_left+(n_right/n)*e_right)
        
        info_gain = parent_entropy - weighted_entropy
        
        return info_gain
        
    def split(self,feature,split_pt):
        true_idx = np.where(feature<=split_pt)[0]
        false_idx = np.where(feature>split_pt)[0]
        return true_idx,false_idx
    
    def bestCriteria(self,X_train,Y_train,feature_idx,feature_list):
        
        best_gain = -1
        best_feature,best_split_pt = None,None
        
        for i in feature_idx:
            if feature_list[i] == False:
                continue
            feature = X_train[:,i]
            sorted_feature = sorted(feature)
            split_pts = []
            
            for j in range(1,len(sorted_feature)):
                pt = sorted_feature[j]/2 + sorted_feature[j-1]/2
                split_pts.append(pt)
                
            for pt in split_pts:
            
                info_gain = self.informationGain(feature,Y_train,pt)
                
                if info_gain>best_gain:
                    best_feature = i
                    best_split_pt = pt
                    best_gain = info_gain
                    
        feature_list[best_feature] = False
    
                    
        return best_feature,best_split_pt,feature_list
    
   
    def fit(self,X_train,Y_train):
        feature_list = np.array([True for i in range(X_train.shape[1])])
        self.root = self.growTree(X_train,Y_train,feature_list)
        print("Tree Built")
        
    def mostCommonValue(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
  
        
    def growTree(self,X_train,Y_train,feature_list,depth = 0):
        n_samples,n_features = X_train.shape 
        n_labels = len(set(Y_train))
        
        if(depth > self.max_depth or n_labels == 1 or n_samples<2 or feature_list.sum()==0):
            leaf_value = self.mostCommonValue(Y_train)
            return Node(value=leaf_value)
        
        feature_idx = np.array([i for i in range(n_features)],dtype = int)
        
        feature_id,split_pt,feature_list = self.bestCriteria(X_train,Y_train,feature_idx,feature_list)
        
        left_idx,right_idx = self.split(X_train[:,feature_id],split_pt)
        
        left = self.growTree(X_train[left_idx,:],Y_train[left_idx],feature_list,depth+1)
        right = self.growTree(X_train[right_idx,:],Y_train[right_idx],feature_list,depth+1)
        
        return Node(left = left,right = right,split_feature_index = feature_id,split_value = split_pt)
        
    
    def predict(self,X_test):
        return np.array([self.traverseTree(x,self.root) for x in X_test])
        
    
    def traverseTree(self,x,node):
        if node.isPure():
            return node.value
        
        if x[node.feature]<=node.split_value:
            return self.traverseTree(x,node.left)
        
        return self.traverseTree(x,node.right)
        
        
        
           
        
        
    
        
