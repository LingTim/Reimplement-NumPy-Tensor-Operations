import pytest
import numpy as np
import my_NDarray_module

# 測試NDarray的建立
def test_NDarray_creation():
    np_array = np.zeros((2, 3, 4))
    my_NDarray = my_NDarray_module.Tensor([2, 3, 4])
    assert my_NDarray.shape == list(np_array.shape)

    np_array2 = np.zeros((2, 2))
    my_NDarray2 = my_NDarray_module.Tensor([2, 2], [1, 2, 3, 4])
    assert my_NDarray2.shape == list(np_array2.shape)

# 測試NDarray和NDarray的逐位元運算
def test_NDarray_by_NDarray():
    shape = [2, 2]
    a = my_NDarray_module.Tensor(shape, 6.0)
    b = my_NDarray_module.Tensor(shape, [1.0, 2.0, 3.0, 4.0])

    # 加法
    c_add = a + b
    assert c_add[[0, 0]] == 7.0

    # 減法
    c_sub = a - b
    assert c_sub[[0, 1]] == 4.0

    # 乘法
    c_mul = a * b
    assert c_mul[[1, 0]] == 18.0

    # 除法
    c_div = a / b
    assert c_div[[1, 1]] == 1.5

# 測試NDarray和Scalar的逐位元運算
def test_NDarray_with_Scalar():
    shape = [2, 2]
    a = my_NDarray_module.Tensor(shape, 10.0)
    
    # 加法
    c_add = a + 5.0
    assert c_add[[0, 0]] == 15.0

    # 減法
    c_sub = a - 2.0
    assert c_sub[[0, 1]] == 8.0

    # 乘法
    c_mul = a * 3.0
    assert c_mul[[1, 0]] == 30.0

    # 除法
    c_div = a / 2.0
    assert c_div[[1, 1]] == 5.0

# 測試NDarray和Scalar的反向逐位元運算
def test_Scalar_with_NDarray():
    shape = [2, 2]
    a = my_NDarray_module.Tensor(shape, [8.0, 7.0, 6.0, 5.0])
    
    # 加法
    c_add = 10.0 + a
    assert c_add[[0, 0]] == 18.0

    # 減法
    c_sub = 10.0 - a
    assert c_sub[[0, 1]] == 3.0

    # 乘法
    c_mul = 2.0 * a
    assert c_mul[[1, 0]] == 12.0

    # 除法
    c_div = 20.0 / a
    assert c_div[[1, 1]] == 4.0

# 測試broadcasting
def test_Broadcasting():
    a = my_NDarray_module.Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    b = my_NDarray_module.Tensor([3], [2, 4, 6])
    c = a + b

    assert c.shape == [2, 3]

    assert c[[0, 0]] == 3
    assert c[[0, 1]] == 6
    assert c[[0, 2]] == 9
    assert c[[1, 0]] == 6
    assert c[[1, 1]] == 9
    assert c[[1, 2]] == 12

# 測試Reduction
def test_Reduction():
    a = my_NDarray_module.Tensor([2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    
    b = a.sum()
    assert b.shape == [1]
    assert b[[0]] == 300

    c = a.sum(0)
    assert c.shape == [3, 4]
    assert c[[0, 0]] == 14

    d = a.sum(1)
    assert d.shape == [2, 4]
    assert d[[0, 0]] == 15

    e = a.sum(2)
    assert e.shape == [2, 3]
    assert e[[0, 0]] == 10

def test_Tensor_Contraction():
    a = my_NDarray_module.Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    b = my_NDarray_module.Tensor([3, 2], [7, 8, 9, 10, 11, 12])
    c = a.matmul(b)

    assert c.shape == [2, 2]
    assert c[[0, 0]] == 58
    assert c[[0, 1]] == 64
    assert c[[1, 0]] == 139
    assert c[[1, 1]] == 154

def test_same_shape_batch_ND_Tensor_Contraction():
    a = my_NDarray_module.Tensor([2, 3, 4], [1.0] * 24)
    b = my_NDarray_module.Tensor([2, 4, 2], [2.0] * 16)
    c = a @ b

    assert c.shape == [2, 3, 2]

    # 驗證 Batch 0
    assert c[[0, 0, 0]] == 8.0
    assert c[[0, 2, 1]] == 8.0
    
    # 驗證 Batch 1
    assert c[[1, 0, 0]] == 8.0
    assert c[[1, 2, 1]] == 8.0

def test_different_shape_batch_ND_Tensor_Contraction():
    a = my_NDarray_module.Tensor([10, 1, 2, 3], [1.0] * 60)
    b = my_NDarray_module.Tensor([5, 3, 4], [2.0] * 60)
    c = a @ b

    assert c.shape == [10, 5, 2, 4]

    assert c[[0, 0, 0, 0]] == 6.0
    assert c[[9, 4, 1, 3]] == 6.0
    assert c[[5, 2, 0, 2]] == 6.0
    assert c[[1, 0, 1, 0]] == 6.0

def test_transpose():
    a = my_NDarray_module.Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    b = a.T
    c = a @ b

    assert c.shape == [2, 2]

    assert c[[0, 0]] == 14.0
    assert c[[0, 1]] == 32.0
    assert c[[1, 0]] == 32.0
    assert c[[1, 1]] == 77.0