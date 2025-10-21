# Python
import os
import builtins
import pytest

from functions import max, work, print_string_to_file, work2

def test_max_basic():
    assert max(1, 2) == 2
    assert max(2, 1) == 2
    assert max(-1, -2) == -1
    assert max(0, 0) == 0
    assert max(2.5, 2.4) == 2.5
    assert max(-5, 5) == 5

def test_work_even_squares():
    assert work([1, 2, 3, 4]) == [4, 16]
    assert work([2, 4, 6]) == [4, 16, 36]
    assert work([1, 3, 5]) == []
    assert work([]) == []
    assert work([-2, -4]) == [4, 16]

def test_work2_prints_sum_of_odd_squares(capsys):
    work2([1, 2, 3, 4])  # 1^2 + 3^2 = 1 + 9 = 10
    captured = capsys.readouterr()
    assert captured.out.strip() == "10"

    work2([2, 4, 6])
    captured = capsys.readouterr()
    assert captured.out.strip() == "0"

    work2([1, 3, 5])
    captured = capsys.readouterr()
    assert captured.out.strip() == str(1*1 + 3*3 + 5*5)

    work2([])
    captured = capsys.readouterr()
    assert captured.out.strip() == "0"

    work2([-1, -3])
    captured = capsys.readouterr()
    assert captured.out.strip() == str((-1)**2 + (-3)**2)