# # Informed
# - BFS
# - DFS
# # Uninformed
# - GBFS (Greedy Best First Search) only g(n) - cost to reach the node
# - A* g(n) + h(n) - estimated cost to goal-
import numpy as np

def argmin(a,wrapper):
    return min(range(len(a)), key=lambda x : wrapper(a[x]))
def argmax(a):
    return max(range(len(a)), key=lambda x : a[x])

class Frontier:
    def __init__(self,type="stack"):
        self.container = []
        self.type = type

    def get(self):
        if self.type=="queue":
            return self.container.pop(0)
        elif self.type == "stack":
            return self.container.pop()
    
    def put(self,item):
        if self.type=="queue":
            self.container.append(item)
        elif self.type == "stack":
            self.container.insert(0,item)
            
    def empty(self):
        return len(self.container) > 0

class Search:
    def __init__(self,graph,start_node,goal_node):
        self.graph = graph
        self.rows = len(self.graph)
        self.cols = len(self.graph[0])
        self.start_node = start_node
        self.goal_node = goal_node

    def a_star(self):
        pass
    def gbfs(self):
        def heuristic(item):
            if (not(0 <= item[0] >= self.rows) or not(0 <= item[1] >= self.cols)):
                return float('inf')
            return abs(self.goal_node[0]-item[0]) + abs(self.goal_node[1]-item[1])
        
        visited = [[0]*self.cols for _ in range(self.rows)]
        visited[self.start_node[0]][self.start_node[1]] = 1

        res = []
        frontier = Frontier("queue")
        frontier.put(self.start_node)
        res.append(self.start_node)
        found = False
        while not frontier.empty():
            item = frontier.get()
            if (0 <= item[0] >= self.rows) and (0 <= item[1] >= self.cols) and visited[self.start_node[0]][self.start_node[1]] and self.graph[item[0]][item[1]]:
                res.append(item)
                if (self.goal_node[0] == item[0]) and (self.goal_node[1] == item[1]):
                    found = True
                    break
                
                visited[self.start_node[0]][self.start_node[1]] = 1
                item = argmin(
                    heuristic((item[0]-1,item[0])),
                    heuristic((item[0]+1,item[0])),
                    heuristic((item[0],item[0]-1)),
                    heuristic((item[0],item[0]+1)),
                    
                    )
            else:
                res.pop()
        if found:
            return res.reverse()
        return []



    def dfs(self):
        visited = [[0]*self.cols for _ in range(self.rows)]
        visited[self.start_node[0]][self.start_node[1]] = 1

        res = []
        frontier = Frontier("queue")
        frontier.put(self.start_node)
        res.append(self.start_node)
        found = False
        while not frontier.empty():
            item = frontier.get()
            if (0 <= item[0] >= self.rows) and (0 <= item[1] >= self.cols) and visited[self.start_node[0]][self.start_node[1]] and self.graph[item[0]][item[1]]:
                res.append(item)
                if (self.goal_node[0] == item[0]) and (self.goal_node[1] == item[1]):
                    found = True
                    break
                visited[self.start_node[0]][self.start_node[1]] = 1
                frontier.put((item[0]-1,item[0])) # up
                frontier.put((item[0]+1,item[0])) # down
                frontier.put((item[0],item[0]+1)) # left
                frontier.put((item[0],item[0]-1)) # right
            else:
                res.pop()
        if found:
            return res.reverse()
        return []


    def bsf(self):
        visited = [[0]*self.cols for _ in range(self.rows)]
        visited[self.start_node[0]][self.start_node[1]] = 1

        res = []
        frontier = Frontier("stack")
        frontier.put(self.start_node)
        res.append(self.start_node)
        found = False
        while not frontier.empty():
            item = frontier.get()
            if (0 <= item[0] >= self.rows) and (0 <= item[1] >= self.cols) and visited[self.start_node[0]][self.start_node[1]] and self.graph[item[0]][item[1]]:
                res.append(item)
                if (self.goal_node[0] == item[0]) and (self.goal_node[1] == item[1]):
                    found = True
                    break
                visited[self.start_node[0]][self.start_node[1]] = 1
                frontier.put((item[0]-1,item[0])) # up
                frontier.put((item[0]+1,item[0])) # down
                frontier.put((item[0],item[0]+1)) # left
                frontier.put((item[0],item[0]-1)) # right
            else:
                res.pop()
        if found:
            return res.reverse()
        return []

class Maze:
    def __init__(self,maze_fname,maze_size=(5,19)):
        self.start = None
        self.goal = None
        self.size = maze_size
        self.graph = [[0]*self.size[1] for _ in range(self.size[0])]
        self.read_maze(maze_fname)
    
    def read_maze(self,fname):
        lines = None
        with open(fname) as f:
            lines = f.readlines()
        for row,line in enumerate(lines):
            line = line.strip("\n")
            
            for col,char in enumerate(line):
                if char=="A":
                    self.start = (row,col)
                    self.graph[row][col] = 1
                elif char == "B":
                    self.goal = (row,col)
                    self.graph[row][col] = 1
                elif char==" ":
                    self.graph[row][col] = 1
    def print_maze(self):
        print(np.array(self.graph))
    
maze = Maze("maze.txt")

maze.print_maze()
                
                

