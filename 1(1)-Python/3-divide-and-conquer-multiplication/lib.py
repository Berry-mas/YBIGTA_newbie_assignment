from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        """
        Matrix 객체를 초기화합니다.

        Args:
            matrix (list[list[int]]): 행렬 데이터
        """
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        """
        모든 원소가 n인 행렬을 생성합니다.

        Args:
            n (int): 행렬의 모든 원소 값
            shape (tuple[int, int]): (행, 열) 형태의 행렬 크기

        Returns:
            Matrix: 생성된 행렬
        """
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        """
        모든 원소가 0인 행렬을 생성합니다.

        Args:
            shape (tuple[int, int]): (행, 열) 형태의 행렬 크기

        Returns:
            Matrix: 생성된 영행렬
        """
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        """
        모든 원소가 1인 행렬을 생성합니다.

        Args:
            shape (tuple[int, int]): (행, 열) 형태의 행렬 크기

        Returns:
            Matrix: 생성된 행렬
        """
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        """
        n x n 단위 행렬을 생성합니다.

        Args:
            n (int): 행과 열의 크기 (정방행렬)

        Returns:
            Matrix: n x n 단위 행렬
        """
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        """
        행렬의 크기를 반환합니다.

        Returns:
            tuple[int, int]: (행 개수, 열 개수)
        """
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        """
        행렬을 깊은 복사한 객체를 생성합니다.

        Returns:
            Matrix: 복사된 새로운 행렬
        """
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        """
        행렬의 특정 위치 값을 반환합니다.

        Args:
            key (tuple[int, int]): (i, j) 인덱스

        Returns:
            int: 해당 위치의 값
        """
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """
        행렬의 특정 위치 값을 설정합니다 (MOD로 나눈 나머지 저장).

        Args:
            key (tuple[int, int]): (i, j) 인덱스
            value (int): 저장할 값
        """
        i, j = key
        self.matrix[i][j] = value % self.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        """
        행렬 곱셈을 수행합니다.

        Args:
            matrix (Matrix): 곱셈 대상 행렬

        Returns:
            Matrix: 곱셈 결과 행렬

        Raises:
            AssertionError: 곱셈이 가능한 행렬이 아닐 경우
        """
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        행렬의 n제곱을 빠른 거듭제곱 방식으로 계산합니다.

        Args:
            n (int): 지수 (양의 정수)

        Returns:
            Matrix: 거듭제곱 결과

        Raises:
            AssertionError: 정방행렬이 아닐 경우
        """
        assert self.shape[0] == self.shape[1]
        result = Matrix.eye(self.shape[0])
        base = self.clone()

        while n > 0:
            if n % 2 == 1:
                result = result.__matmul__(base)
            base = base.__matmul__(base)
            n //= 2

        return result


    def __repr__(self) -> str:
        """
        행렬을 문자열로 반환합니다.

        Returns:
            str: 각 행을 한 줄씩 출력하는 형태의 문자열
        """
        return '\n'.join(' '.join(str(cell % self.MOD) for cell in row) for row in self.matrix)