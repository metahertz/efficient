def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


class Calculator:
    def __init__(self, initial: int = 0):
        self.value = initial

    def add(self, n: int) -> "Calculator":
        self.value += n
        return self

    def result(self) -> int:
        return self.value
