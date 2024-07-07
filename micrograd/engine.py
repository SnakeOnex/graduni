# questions:
# Q: why is _prev a set of _children?
# A:
# Q: 

from __future__ import annotations
from typing import Union

# Backpropagation
# ---------------
# For backprop, each Value object needs to remember the computation that preceded it,
# so that gradients can be computed all the way to the source

# x = a + b
# dx/da = 1 & dx/db = 1

class Value:
    def __init__(self, data: float, _parents=[]):
        self.data = data
    def __add__(self, other: Union[Value, float]):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data)
    def __mul__(self, other: Union[Value, float]):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data)
    def __repr__(self):
        return f"Value({self.data})"


if __name__ == "__main__":
    a = Value(3.5)
    b = Value(2.)

    print(f"a={a}, b={b}")
    print(f"a+b = {a+b}")
    print(f"a*b = {a*b}")
