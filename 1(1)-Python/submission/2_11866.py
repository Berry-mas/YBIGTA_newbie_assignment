from __future__ import annotations
from collections import deque


"""
TODO:
- rotate_and_remove 구현하기 
"""


def create_circular_queue(n: int) -> deque[int]:
    """1부터 n까지의 숫자로 deque를 생성합니다."""
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환합니다.

    Args : 
        queue (deque[int]): 메서드 대상 큐
        k (int) : 제거 대상 번호

    Returns: 
        int: 제거되고 반환된 원소소
    """
    queue.rotate(-(k-1))
    result_int = queue.popleft()
    return result_int




"""
TODO:
- josephus_problem 구현하기
    # 요세푸스 문제 구현
        # 1. 큐 생성
        # 2. 큐가 빌 때까지 반복
        # 3. 제거 순서 리스트 반환
"""


def josephus_problem(n: int, k: int) -> list[int]:
    """
    요세푸스 문제 해결
    n명 중 k번째마다 제거하는 순서를 반환

    Args:
        - n(int) : 전체 사람 수
        - k(int) : 제거 번호

    Returns :
        - list[int] : 제거된 번호의 순서 리스트트
    """
    queue = create_circular_queue(n)
    result_list = []
    while len(queue) > 0 :
        result_list.append(rotate_and_remove(queue, k))
    
    return result_list


def solve_josephus() -> None:
    """입, 출력 format"""
    n: int
    k: int
    n, k = map(int, input().split())
    result: list[int] = josephus_problem(n, k)
    
    # 출력 형식: <3, 6, 2, 7, 5, 1, 4>
    print("<" + ", ".join(map(str, result)) + ">")

if __name__ == "__main__":
    solve_josephus()