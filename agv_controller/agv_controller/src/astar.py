from math import sqrt
class map:
    """
    [
    [0,0,0,0]
    [0,0,1,0]
    ]
    """
    def __init__(self, grid_occupancy_map, dim):
        self.map = grid_occupancy_map
        self.xdim = dim[0]
        self.ydim = dim[1]

    def check_map(self):
        for i in self.map:
            for j in i:
                if j != 1 and j != 0:
                    return False
        return True
    
    def __euclid_dist(self,a,b):
        A = (a[0]-b[0])**2
        B = (a[1]-b[1])**2
        ed = sqrt(A+B)
        md = abs(a[0] - b[0]) + abs(a[1]-b[1])
        return ed

    def __is_valid_node(self, node, xmax, ymax):
        if 0 <= node[0] < ymax and 0 <= node[1] < xmax:
            return True
        else:
            return False

    def astar(self, start, end):
        if start[0] >= self.ydim or start[1] >= self.xdim or end[0] >= self.ydim or end[1] >= self.xdim:
            raise Exception("The start or end point is out of bounds")
        
        if start[0] < 0 or start[1] < 0 or end[0] < 0 or end[1] < 0:
            raise Exception("Coordinates can not be negative.")

        if self.map[start[0]][start[1]] == 1 or self.map[end[0]][end[1]] == 1:
            raise Exception("The starting node or the ending node is unreachable")
        
        
        open = []
        closed = []
        dist_to_goal = self.__euclid_dist(start, end)
        temp = [start, 0, dist_to_goal, dist_to_goal, start] #formated as node, G cost(dist from start), H cost(Dist from goal), F cost, parent node
        print(temp)
        open.append(temp)
        iter = 0
        while(len(closed) < self.xdim*self.ydim ):
            iter += 1
            #print(iter)
            if(len(open) < 1):
                raise Exception("No Path Exists")
            
            min_f = open[0][3]
            idx = 0
            for i in range(len(open)): #To find the node with the lowerst F value
                if  min_f > open[i][3]:
                    idx = i
                    min_f = open[i][3]
            current = open.pop(idx)
            closed.append(current)
            
            if current[0] == end:
                return closed
            
            #curret node, each node is formatted as (y,x)
            up = (current[0][0]-1,  current[0][1])
            down = (current[0][0]+1,  current[0][1])
            left = (current[0][0],  current[0][1]-1)
            right = (current[0][0],  current[0][1]+1)
            ul = (current[0][0]-1,  current[0][1]-1)
            ur = (current[0][0]-1,  current[0][1]+1)
            ll = (current[0][0]+1,  current[0][1]-1)
            lr = (current[0][0]+1,  current[0][1]+1)
            adjacent_nodes = [up, down, left, right, ul, ur, ll, lr]
            for i in range(len(adjacent_nodes)):
                if(self.__is_valid_node(adjacent_nodes[i], self.xdim, self.ydim)):
                    #print(f"Valid Node: {temparr[i]}")
                    if self.map[adjacent_nodes[i][0]][adjacent_nodes[i][1]] == 1:
                        continue
                    c = False #Assume node is not closed
                    #Check if node is closed
                    for j in closed:
                        if j[0] == adjacent_nodes[i]:
                            c = True
                            break
                    #If node is closed skip to next node
                    if c:
                        continue
                    #Calculate the g,h and f value for the new node
                    h = self.__euclid_dist(adjacent_nodes[i], end)
                    g = current[1]
                    #If node can be reached by a straight line add 10, if diagonal add 14
                    if(i<4):
                        g += 10
                    else:
                        g += 14
                    f = g+h
                    #create the list with all the data about the node 
                    t = [adjacent_nodes[i], g, h, f, current]
                    present_in_open = False
                    #Check if node is already opened, if so then update to shortest path to reach the node
                    for open_node in open:
                        #print(f"{open_node[0]} == {temparr[i]} ?? " )
                        if open_node[0] == adjacent_nodes[i]:
                            if open_node[1] > g:
                                open_node[1] = g
                                open_node[3] = g + open_node[2]
                                open_node[4] = current
                            present_in_open = True
                            break
                    
                    if not present_in_open:
                        open.append(t)
                    else:
                        continue