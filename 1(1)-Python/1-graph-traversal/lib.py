from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    """
    DFS와 BFS 탐색 기능을 제공하는 클래스입니다.

    Attributes:
        n (int): 그래프의 정점 수
        graph (dict[int, list[int]]): 각 정점에 연결된 인접 정점 리스트를 저장하는 딕셔너리

    Methods:
        add_edge(u, v): 정점 u와 v 사이에 간선 추가가
        dfs(v, visited): 깊이 우선 탐색을 수행행
        bfs(v): 너비 우선 탐색을 수행합니다.
        search_and_print(start): 시작 정점에서 DFS, BFS 순으로 탐색하고 결과 출력력
    """
    def __init__(self, n: int) -> None:
        """
        1~n까지의 각 정점 번호를 key로,
        빈 리스트를 value로 갖는 딕셔너리를 통해 그래프를 초기화합니다.
        
        Args:
            - n (int): 정점의 개수 (1번부터 n번까지)
        """
        self.n = n
        self.graph: DefaultDict[int, List[int]] = defaultdict(list)
        for i in range(1, n+1) : 
            self.graph[i] = []

    
    def add_edge(self, u: int, v: int) -> None:
        """
        정점 사이에 양방향 간선을 추가합니다.

        Args:
            - u(int): 정점 u
            - v (int): 정점 v
        """
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)을 수행합니다.
        
        구현 방법 선택:
        1. 재귀 방식: 함수 내부에서 재귀 함수 정의하여 구현
        2. 스택 방식: 명시적 스택을 사용하여 반복문으로 구현

        Args:
            - start(int): 시작 노드 번호

        Returns: 
            - list[int] : 방문한 노드의 순서를 담은 방문 배열
        """
        visit = [False]*(self.n+1)
        result_list = []

        def dfs_inside(v: int) -> None:
            visit[v] = True
            result_list.append(v)
            for neighbor in sorted(self.graph[v]):
                if visit[neighbor] == False :
                    dfs_inside(neighbor)

        dfs_inside(start)
        
        return result_list
        
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)을 수행합니다.

        큐를 사용하여 구현

        Args:
            - start(int): 시작 노드 번호

        Returns: 
            - list[int] : 방문한 노드의 순서를 담은 방문 배열
        """
        visit = [False]*(self.n+1)
        result_list = []
        visit[start] = True
        queue = deque([start])

        while queue:
            v = queue.popleft()
            result_list.append(v)
            for neighbor in sorted(self.graph[v]):
                if visit[neighbor] == False :
                    visit[neighbor] = True
                    queue.append(neighbor)
        
        return result_list
            

    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
