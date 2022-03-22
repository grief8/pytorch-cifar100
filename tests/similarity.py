import math

import numpy as np
import scipy.stats


def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
    注意有展平操作。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:实际是夹角的余弦值，ret = (cos+1)/2
    """
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(farr2 ** 2))
    similar = numer / denom  # 这实际是夹角的余弦值
    return (similar + 1) / 2  # 姑且把余弦函数当线性


def mtx_similar2(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    计算对矩阵1的相似度。相减之后对元素取平方再求和。因为如果越相似那么为0的会越多。
    如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    """
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0], arr2.shape[0])
        miny = min(arr1.shape[1], arr2.shape[1])
        differ = arr1[:minx, :miny] - arr2[:minx, :miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ ** 2)
    denom = np.sum(arr1 ** 2)
    similar = 1 - (numera / denom)
    return similar


def mtx_similar3(arr1: np.ndarray, arr2: np.ndarray) -> float:
    '''
    From CS231n: There are many ways to decide whether
    two matrices are similar; one of the simplest is the Frobenius norm. In case
    you haven't seen it before, the Frobenius norm of two matrices is the square
    root of the squared sum of differences of all elements; in other words, reshape
    the matrices into vectors and compute the Euclidean distance between them.
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0], arr2.shape[0])
        miny = min(arr1.shape[1], arr2.shape[1])
        differ = arr1[:minx, :miny] - arr2[:minx, :miny]
    else:
        differ = arr1 - arr2
    dist = np.linalg.norm(differ, ord=None)
    len1 = np.linalg.norm(arr1)
    len2 = np.linalg.norm(arr2)  # 普通模长
    denom = (len1 + len2) / 2
    similar = 1 - (dist / denom)
    return similar


# def wasstein
def test_similar():
    arr1 = np.array([[1, -2, 3, 7], [-8, 2, 5, 9]])
    arr2 = np.array([[1, -2, 3, 7], [-8, 2, 6, 9]])
    arr3 = np.array([[-2, 3, 7], [2, 7, 9]])
    arr4 = np.array([[4, -2, 3], [-8, 2, 7]])
    print('similar arr1&2:', mtx_similar1(arr1, arr2),
          mtx_similar2(arr1, arr2), mtx_similar3(arr1, arr2), sep=' ')
    print('similar arr2&3:', mtx_similar1(arr2, arr3),
          mtx_similar2(arr2, arr3), mtx_similar3(arr2, arr3), sep=' ')
    print('similar arr2&4:', mtx_similar1(arr2, arr4),
          mtx_similar2(arr2, arr4), mtx_similar3(arr2, arr4), sep=' ')
    print('similar arr4&4:', mtx_similar1(arr4, arr4),
          mtx_similar2(arr4, arr4), mtx_similar3(arr4, arr4), sep=' ')


def kl_divergence(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    std = max(abs(x.min()), abs(y.min()))
    x = x + std + 1
    y = y + std + 1
    # px = x / np.sum(x)
    # py = y / np.sum(y)
    # # 编程实现
    # KL = 0.0
    # for i in range(len(x)):
    #     KL += px[i] * np.log(px[i] / py[i])
        # KL = max(KL, x[i] * np.log(x[i] / y[i]))
    # 利用scipy API进行计算
    # scipy计算函数可以处理非归一化情况，因此这里使用
    # scipy.stats.entropy(x, y)或scipy.stats.entropy(px, py)均可
    KL = scipy.stats.entropy(x, y)

    return KL