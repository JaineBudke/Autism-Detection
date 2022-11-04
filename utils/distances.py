import math

class Distances:

    # calculate manhattan distance
    # p1, p2: lists represent points on a multidimensional plane 
    def manhattanDistance( p1, p2 ):
        
        result = 0.0
        
        # lopp on p1 values
        for i in range(len(p1)):

            result += abs(p1[i] - p2[i])
            
        return result

    
    # calculate euclidean distance
    # p1, p2: lists represent points on a multidimensional plane 
    def euclideanDistance( p1, p2 ):
        result = 0.0
        for i in range(len(p1)):

            result += ((p1[i] - p2[i])**2)
        
        return math.sqrt(result)


    def cosineSimilarity( p1, p2 ):

        p1p2 = 0.0
        p1a2 = 0.0
        p2a2 = 0.0
        for i in range(len(p1)):
            p1p2 += (p1[i] * p2[i])
            p1a2 += (p1[i] ** 2)
            p2a2 += (p2[i] ** 2)
        
        return p1p2/( math.sqrt(p1a2) * math.sqrt(p2a2) )
