# 矩阵LU分解

## 问题分析

LU分解的串行算法如下，分析数据依赖关系发现，第一个`for`循环不可并行，第一个`for`循环内的两个`for`循环可以合成一个`for`循环，且可以并行

```
LU(Mat A):
	// 不可并行
    for k = 1 to n:
    	// 以下两个循环可以合为一个循环，且可以并行
        for i = k + 1 to n:
            A[i][k] = A[i][k]/A[k][k]
        for i = k + 1 to n:
            for j = k + 1 to n:
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
                
    Mat L, U
    for i = 1 to n:
        for j = 1 to n:
            if i > j:
                L[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]
        L[i][i] = 1

```

并行后伪代码如下

```
LU(Mat A):
	#pragma omp parallel
    {
    for k = 1 to n:
	    #pragma omp for
        for i = k + 1 to n:
            A[i][k] = A[i][k]/A[k][k]
            for j = k + 1 to n:
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
    }
    
    Mat L, U
    for i = 1 to n:
        for j = 1 to n:
            if i > j:
                L[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]
        L[i][i] = 1

```

并行域需要包括外围`for`循环，否则openmp将在每一个`for`循环内反复创建销毁进程，造成额外开销

## 实验结果

| 线程数 | 执行时间（秒） |
| ------ | -------------- |
| 1      | 7.4            |
| 2      | 4.8            |
| 4      | 3.5            |
| 8      | 3.1            |
| 16     | 3.2            |
| 32     | 3.8            |

可见随着线程数的增加，执行时间先缩短再上升。但是当线程数较少时，程序也无法达到线性的加速比

这是因为LU分解任务无法并行化外围循环，只能并行化内部循环，而且内部循环随着外部循环的进行，运算量逐渐减少。因此程序运行前期，并行加速的同时线程同步的开销并不明显，加速比较高，而程序运行后期，线程同步变得频繁，加速比降低

当线程数过高时，线程同步的开销增长速度超过了并行的加速，造成加速比降低

## 结论

并行化外围循环无法并行的任务效果不太好，很难达到客观的加速比

用openmp并行化内层循环时，应该让并行域包括外层循环，避免反复创建销毁线程的开销

# 单源最小路径问题

## 问题分析

delta-stepping算法的伪代码如下

![img](https://github.com/zzy99/delta-stepping/raw/master/02.png)

可以将该算法简单地概括为以下步骤

1. 找到第一个非空的桶，找到其中顶点对应的轻边（权重小于$\Delta$），松弛之，保存被操作过的顶点
2. 第一步中松弛的步骤可能为该桶引入新的顶点。若桶非空，则再次执行第一步
3. 当桶空后，找到之前保存的被操作过的顶点，找到其对应的重边（权重大于$\Delta$），松弛之
4. 第三步不会为该桶引入新的顶点，向后寻找下一个非空桶，重复第一步

其中第一步、第三步本身可以直接并行化，但是第一步执行完后，需要检查是否有新顶点出现，无法直接并行化，因此当$\Delta$较大时，第一步和第二步需要反复多次，其中隐含多个线程同步操作，并行收益低

当$\Delta$较小时，第一步和第二步执行次数少，主要执行第三步，算法逐渐退化为Bellman-Ford算法

在我的实现中，我试图使所有的桶能够并行，步骤如下所示

1. 根据记录的最小桶索引、最大桶索引，并行地松弛所有桶中顶点对应的边
2. 若需要更新桶和距离数组，使用 `omp critical`阻塞更新，此时桶中可能出现新的顶点，最大桶索引将会增加
3. 当所有桶都为空时（实际上由于判断语句和阻塞更新的先后顺序，所有桶不一定都为空），重复第一步

实际上，从结果可以看出这样的并行方法效果并不好

## 实验结果

| 线程数 | 执行时间（秒） |
| ------ | -------------- |
| 1      | 7.7            |
| 2      | 7.8            |
| 4      | 11.1           |
| 8      | 11.0           |
| 16     | 12.8           |

运行时间随线程数增加逐渐增加，这是因为每次并行地松弛时，都需要阻塞区来保证数据一致性，导致当线程数增加时，线程通信的开销非常高。实际上，当线程数为1，$\Delta$非常小（个位数）时，即桶的个数非常多时，程序达到最优性能。

## 结论

我的程序并行度不够好，但是串行就能打过助教的参考程序，可能也说明该问题的并行化不太容易

# K-means聚类

## 问题分析

K-means串行算法如下

```
K_means():
    read N*M-dimension datas
    randomly generate K points as centers
    // 不能并行
    while iteration < max:
    	// 可以并行
        for data in datas:
            for center in cluster centers:
                calculate distance of data and center
            determine center that data belongs to
        for center in cluster centers:
            calculate new coordinate of center

```

分析可知`while`循环迭代存在前后数据依赖，不能并行，`while`循环内的内容则可以通过将数据`datas`均分给不同的进程并行执行，并行版本的伪代码如下

```
K_means_parallel():

    如果是主进程:
        读取输入N, M, K
    广播N, M, K到所有进程（MPI_Bcast）

    如果是主进程:
        读取N*M维度的数据datas
        随机生成K个中心点centers

    均分数据datas，发送给对应进程（MPI_Scatterv）
    广播中心点centers到所有进程（MPI_Bcast）

    while iteration < max:
        对于每个进程的本地数据点:
            对于每个中心点:
                计算数据点与中心点的距离
            确定数据点所属的中心点
        对于每个中心点:
            计算新的聚类坐标之和

        汇总所有进程的聚类坐标之和和聚类大小（MPI_Allreduce）
        根据整体的聚类坐标之和和聚类大小，更新中心点坐标


```

为了可减少MPI中的通信开销，我将所有的`MPI_Send` `MPI_Receive`函数都改为了其他的方法，例如均分数据点时使用`MPI_Scatterv`，广播中心点时使用`MPI_Bcast`，汇总聚类坐标和时使用`MPI_Allreduce`，减少了通信次数和阻塞次数，提高了通信效率，同时也提高了代码的可读性



## 实验结果

MPI程序无法在代码层面控制进程数，无法在OJ系统上控制运行使用的进程数，因此无法通过OJ系统来测量程序的可扩放性，因此在我自己的电脑上测量，结果如下

| 进程数 | 执行时间（秒） |
| ------ | -------------- |
| 1      | 0.13           |
| 2      | 0.25           |
| 3      | 0.38           |
| 4      | 0.50           |
| 5      | 0.61           |
| 6      | 0.73           |
| 7      | 0.86           |
| 8      | 1.01           |

可见程序的可扩放性并不好，这可能是由于输入数据的规模太小，只有200个7维数据点，因此进程间通信所消耗的时间远远大于并行计算节省的时间



## 结论

程序的可扩放性和输入数据的规模也有关

MPI程序中可以尽量不使用基础的通信函数，尽量不使用阻塞通信，提升通信效率

# 稀疏矩阵乘法

## 问题分析

稠密矩阵与稀疏矩阵的乘法中，对于结果矩阵中的每一个元素，需要的数据为稠密矩阵的一行和稀疏矩阵的一列，因此对稠密矩阵使用行主序存储，稀疏矩阵按列分开存储，提高内存访问的局部性

其中稀疏矩阵的一列中，非零元素的个数是不确定的，因此若采用最简单的思路，让每一个进程分别计算结果矩阵中的一个元素，会导致每个进程的工作量分配不均衡，降低性能，改进的方法有以下几种

1. 根据稀疏矩阵每一列的元素个数，调整每个进程负责的列数，例如设置阈值为`thr`，当系数矩阵的`b`列的非零元素数大于`thr`，`b+1, b+2, b+3`三列的非零元素数加起来才大于`thr`，则一进程负责结果矩阵的`result[a][b]`，二进程负责结果矩阵的`result[a][b+1], result[a][b+2],result[a][b+3],`
2. 将稀疏矩阵的非零元素总个数均分给所有的进程，所有的进程根据非零元素的行号、列号读取稠密矩阵中的数据进行计算，使用原子加法计算结果矩阵

其中第二种方法在内存访问的局部性上、原子操作的开销上都不如第一种方法



## 实验结果

由于时间原因，我没有采用以上的优化方法，直接使用了最简单的方法，即让每一个进程分别计算结果矩阵中的一个元素，OJ运行时间为5.4秒



## 结论

并行程序的运行时间取决于最慢的进程，均衡不同进程的工作量非常重要

