'''
Created on 3 May 2016

@author: af
'''
import copy
import numpy as np


import pdb

class KDTree:
    def __init__(self, bucket_size, dimensions, parent=None):
        self.bucket_size = bucket_size
        self.parent = None
        self.left = None
        self.right = None
        self.split_dimension = None
        self.split_value = None
        self.index_locations = []
        self.location_count = 0
        self.min_limit = [np.Inf] * dimensions 
        self.max_limit = [-np.Inf] * dimensions
        self.dimensions = dimensions
    
    def get_leaf(self, location):
        if not self.left and not self.right:
            return self
        elif location[self.split_dimension] <= self.split_value:
            return self.left.get_leaf(location)
        else:
            return self.right.get_leaf(location) 
    
    def add_point(self, index_location_tuple):
        self.index_locations.append(index_location_tuple)
        self.location_count += 1
        self.extendBounds(index_location_tuple[1])
        self.min_boundary = copy.deepcopy(self.min_limit)
        self.max_boundary = copy.deepcopy(self.max_limit)
        
    def extendBounds(self, location):
        #empty
        if self.min_limit == None:
            self.min_limit = copy.deepcopy(location)
            self.max_limit = copy.deepcopy(location)
            return
        for i in range(self.dimensions):
            self.min_limit[i] = min(self.min_limit[i], location[i])
            self.max_limit[i] = max(self.max_limit[i], location[i])
    
    def findWidestAxis(self):
        widths = [self.max_limit[i] - self.min_limit[i] for i in range(self.dimensions)]
        widest_axis = np.argmax(widths)
        return widest_axis
    def getNodes(self):
        nodes = []
        self.getNodesHelper(nodes)
        return nodes
    
    def getNodesHelper(self, nodes):
        nodes.append(self)
        if self.left:
            self.left.getNodesHelper(nodes)
        if self.right:
            self.right.getNodesHelper(nodes)
    
    def getLeaves(self):
        leaves = []
        self.getLeavesHelper(leaves)
        return leaves
    
    def getLeavesHelper(self, leaves):
        if not self.right and not self.left:
            leaves.append(self)
        else:
            if self.left:
                self.left.getLeavesHelper(leaves)
            if self.right:
                self.right.getLeavesHelper(leaves)
                
    def balance(self):
        self.nodeSplit(self)
    
    def nodeSplit(self, cursor, empty_non_leaf=True):
        if cursor.location_count > cursor.bucket_size:
            cursor.split_dimension = cursor.findWidestAxis()
            #the partition method is the median of all values in the widest dimension
            cursor.split_value = np.median([cursor.index_locations[i][1][cursor.split_dimension] for i in range(cursor.location_count)])
            # if width is 0 (all the values are the same) don't partition
            if cursor.min_limit[cursor.split_dimension] == cursor.max_limit[cursor.split_dimension]:
                return
            # Don't let the split value be the same as the upper value as
            # can happen due to rounding errors!
            if cursor.split_value == cursor.max_limit[cursor.split_dimension]:
                cursor.split_value = cursor.min_limit[cursor.split_dimension]
            cursor.left = KDTree(bucket_size=cursor.bucket_size, dimensions=cursor.dimensions, parent=cursor)
            cursor.right = KDTree(bucket_size=cursor.bucket_size, dimensions=cursor.dimensions, parent=cursor)
            
            cursor.left.min_boundary = copy.deepcopy(cursor.min_boundary)
            cursor.left.max_boundary = copy.deepcopy(cursor.max_boundary)
            cursor.right.min_boundary = copy.deepcopy(cursor.min_boundary)
            cursor.right.max_boundary = copy.deepcopy(cursor.max_boundary)
            cursor.left.max_boundary[cursor.split_dimension] = cursor.split_value
            cursor.right.min_boundary[cursor.split_dimension] = cursor.split_value
            
            for index_loc in cursor.index_locations:
                if index_loc[1][cursor.split_dimension] > cursor.split_value:
                    cursor.right.index_locations.append(index_loc)
                    cursor.right.location_count += 1
                    cursor.right.extendBounds(index_loc[1])
                else:
                    cursor.left.index_locations.append(index_loc)
                    cursor.left.location_count += 1
                    cursor.left.extendBounds(index_loc[1])
            if empty_non_leaf:
                cursor.index_locations = []
            cursor.nodeSplit(cursor.left)
            cursor.nodeSplit(cursor.right)                    

        
class KDTreeClustering():
    def __init__(self, bucket_size=10):
        self.bucket_size = bucket_size
        self.is_fitted = False
        
    def fit(self, X):
        #X is an array
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
            dimensions = X.shape[1]
        else:
            n_samples = len(X)
            dimensions = len(X[0])
        
        self.kdtree = KDTree(bucket_size=self.bucket_size, dimensions=dimensions, parent=None)
        for i in range(n_samples):
            self.kdtree.add_point((i, X[i]))
        self.kdtree.nodeSplit(cursor=self.kdtree, empty_non_leaf=True)
        self.clusters = [leave.index_locations for leave in self.kdtree.getLeaves()]
        clusters = [cluster.index_locations for cluster in self.kdtree.getLeaves()]
        results = np.zeros((n_samples,), dtype=int)
        for i, id_locs in enumerate(clusters):
            for id, l in id_locs:
                results[id] = i
        self.clusters = results
        self.num_clusters = len(clusters)
        self.is_fitted = True
           
    def get_clusters(self):
        if self.is_fitted:
            return self.clusters
                
if __name__ == '__main__':
    #tree = KDTree(300, 2)
    import params
    import geolocate
    geolocate.initialize(granularity=params.BUCKET_SIZE, write=False, readText=True, reload_init=False, regression=False)
    locations = [geolocate.locationStr2Float(loc) for loc in params.trainUsers.values()]
    clusterer = KDTreeClustering(bucket_size=params.BUCKET_SIZE)
    clusterer.fit(locations)
    clusters = clusterer.get_clusters()
    
    pdb.set_trace()
               
        
