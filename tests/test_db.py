import pytest
from app import create_con

def test_engine_con():
    c = 0
    for _ in create_con():
        c+=1
    print(c)
    assert 0
