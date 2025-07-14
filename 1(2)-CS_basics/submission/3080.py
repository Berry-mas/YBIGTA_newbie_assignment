from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable [T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        node = 0
        for ch in seq:
            found = False
            for child in self[node].children:
                if self[child].body == ch:
                    node = child
                    found = True
                    break

            if not found:
                new_node_index = self.size()
                self.append(TrieNode(body=ch))
                self[node].children.append(new_node_index)
                node = new_node_index

        self[node].is_end = True
    
    def size(self) -> int:
        return len(self)



import sys


"""
TODO:
- 일단 lib.py의 Trie Class부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    input = sys.stdin.readline
    N = int(input())
    names = [input().strip() for _ in range(N)]

    trie = Trie[int]()

    for name in names:
        trie.push(map(ord, name))  # str -> int로 바꿔서 push

    names.sort()

    def get_common_prefix_len(a: str, b: str) -> int:
        l = min(len(a), len(b))
        for i in range(l):
            if a[i] != b[i]:
                return i
        return l

    max_len = 0
    for i in range(N - 1):
        max_len = max(max_len, get_common_prefix_len(names[i], names[i + 1]))

    print(max_len)


if __name__ == "__main__":
    main()