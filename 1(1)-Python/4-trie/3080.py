from lib import Trie
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