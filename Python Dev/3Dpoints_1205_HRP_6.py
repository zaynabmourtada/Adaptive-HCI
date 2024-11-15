'''
Descripttion: 
version: 
Author: 输入自己姓名
Date: 2022-12-07 10:02:52
LastEditors: 输入自己姓名
LastEditTime: 2022-12-07 10:24:07
'''
'''
Descripttion: 
version: 
Author: 输入自己姓名
Date: 2022-12-05 19:03:24
LastEditors: 输入自己姓名
LastEditTime: 2022-12-05 22:27:05
'''
'''
Descripttion: 1205 input the xyz of 6 key points, parse the other 15 key points
version: 
Author: Xiao Zhang
Date: 2022-12-05 16:25:04
LastEditTime: 2022-12-05 19:02:26

'''





from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from scipy.optimize import fsolve

# the 3D info of each point with time variation



# xS = [[0,4.6,2.7,0.3,-2.6,-4.0],[0,4.6,2.7,0.3,-2.6,-4.0],[0,4.6,2.7,0.3,-2.6,-4.0]]
# yS = [[0,10.5,11,14.2,14,11.9],[0,10.5,11,14.2,14,11.9],[0,10.5,11,14.2,14,11.9]]
# zS = [[0,3.8,-5.2,0,6.6,8.1],[0,3.8,-5.2,0,6.6,8.1],[0,3.8,-5.2,0,6.6,8.1]]

# G1
# xS = [[10,15,13,10,7,6],[10,15,13,10,7,6],[10,15,13,10,7,6]]
# yS = [[10,21,21,24,24,22],[10,21,21,24,24,22],[10,21,21,24,24,22]]
# zS = [[10,14,5,10,17,18],[10,14,5,10,17,18],[10,14,5,10,17,18]]

# G2
# xS = [[10,12.87,11.22,12.62,10.84,9.11],[10,12.87,11.22,12.62,10.84,9.11],[10,12.87,11.22,12.62,10.84,9.11]] 
# yS = [[10,13.00,11.28,12.74,10.88,9.07],[10,13.00,11.28,12.74,10.88,9.07],[10,13.00,11.28,12.74,10.88,9.07]] 
# zS = [[10,12.44,13.75,7.78,8.86,11.19],[10,12.44,13.75,7.78,8.86,11.19],[10,12.44,13.75,7.78,8.86,11.19]] 




# G3
# xS = [[10.00,14.51,13.49,12.08,10.27,9.07],[10.00,14.51,13.49,12.08,10.27,9.07],[10.00,14.51,13.49,12.08,10.27,9.07]] 
# yS = [[10.00,18.62,13.49,14.29,14.82,16.32],[10.00,18.62,13.49,14.29,14.82,16.32],[10.00,18.62,13.49,14.29,14.82,16.32]] 
# zS = [[10.00,12.44,11.19,11.19,11.19,11.19],[10.00,12.44,11.19,11.19,11.19,11.19],[10.00,12.44,11.19,11.19,11.19,11.19]] 


# G4
# xS = [[10.00,15.04,12.78,11.24,10.57,7.57],[10.00,15.04,12.78,11.24,10.57,7.57],[10.00,15.04,12.78,11.24,10.57,7.57]] 
# yS = [[10.00,19.85,19.15,24.10,24.01,22.28],[10.00,19.85,19.15,24.10,24.01,22.28],[10.00,19.85,19.15,24.10,24.01,22.28]] 
# zS = [[10.00,11.19,11.19,11.19,12.44,15.13],[10.00,11.19,11.19,11.19,12.44,15.13],[10.00,11.19,11.19,11.19,12.44,15.13]] 



# G5
# xS = [[10.00,12.83,11.33,10.88,8.45,7.22],[10.00,12.83,11.33,10.88,8.45,7.22],[10.00,12.83,11.33,10.88,8.45,7.22]] 
# yS = [[10.00,19.10,19.50,23.26,23.48,22.06],[10.00,19.10,19.50,23.26,23.48,22.06],[10.00,19.10,19.50,23.26,23.48,22.06]] 
# zS = [[10.00,11.19,11.19,16.58,16.58,16.58],[10.00,11.19,11.19,16.58,16.58,16.58],[10.00,11.19,11.19,16.58,16.58,16.58]] 






# G6
# xS = [[10.00,12.03,11.68,10.09,10.75,7.53],[10.00,12.03,11.68,10.09,10.75,7.53],[10.00,12.03,11.68,10.09,10.75,7.53]] 
# yS = [[10.00,18.04,23.34,23.92,17.82,21.62],[10.00,18.04,23.34,23.92,17.82,21.62],[10.00,18.04,23.34,23.92,17.82,21.62]] 
# zS = [[10.00,11.19,16.58,16.58,10.00,16.58],[10.00,11.19,16.58,16.58,10.00,16.58],[10.00,11.19,16.58,16.58,10.00,16.58]] 






# G7

# xS = [[10.00,10.57,12.12,10.62,9.16,9.34],[10.00,10.57,12.12,10.62,9.16,9.34],[10.00,10.57,12.12,10.62,9.16,9.34]] 
# yS = [[10.00,17.64,23.39,23.61,22.59,16.54],[10.00,17.64,23.39,23.61,22.59,16.54],[10.00,17.64,23.39,23.61,22.59,16.54]] 
# zS = [[10.00,8.86,16.58,16.58,16.58,8.86],[10.00,8.86,16.58,16.58,16.58,8.86],[10.00,8.86,16.58,16.58,16.58,8.86]] 






# G8

# xS = [[10.00,13.36,10.40,9.73,8.50,7.08],[10.00,13.36,10.40,9.73,8.50,7.08],[10.00,13.36,10.40,9.73,8.50,7.08]] 
# yS = [[10.00,18.97,24.14,24.85,24.23,22.73],[10.00,18.97,24.14,24.85,24.23,22.73],[10.00,18.97,24.14,24.85,24.23,22.73]] 
# zS = [[10.00,15.13,21.43,19.72,18.11,13.75],[10.00,15.13,21.43,19.72,18.11,13.75],[10.00,15.13,21.43,19.72,18.11,13.75]] 

# G9

# xS = [[10.00,13.37,12.47,11.00,9.67,9.76],[10.00,13.37,12.47,11.00,9.67,9.76],[10.00,13.37,12.47,11.00,9.67,9.76]] 
# yS = [[10.00,19.41,24.82,25.01,24.35,22.49],[10.00,19.41,24.82,25.01,24.35,22.49],[10.00,19.41,24.82,25.01,24.35,22.49]] 
# zS = [[10.00,8.78,15.27,11.50,11.59,13.46],[10.00,8.78,15.27,11.50,11.59,13.46],[10.00,8.78,15.27,11.50,11.59,13.46]] 



# G10

# xS = [[10.00,15.61,12.31,10.63,9.14,8.15],[10.00,15.61,12.31,10.63,9.14,8.15],[10.00,15.61,12.31,10.63,9.14,8.15]] 
# yS = [[10.00,19.41,24.34,24.97,24.52,22.85],[10.00,19.41,24.34,24.97,24.52,22.85],[10.00,19.41,24.34,24.97,24.52,22.85]] 
# zS = [[10.00,13.85,13.85,13.85,13.85,13.85],[10.00,13.85,13.85,13.85,13.85,13.85],[10.00,13.85,13.85,13.85,13.85,13.85]] 


plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
        


while(True):

    

    for c, m, zlow, zhigh in [('r', 'o', 0, 20)]:
        for i in range(len(xS)):

            
            # 这个不能用
            # plt.clf()

            # 清空画布
            ax.cla()

            start = time.time()
            ax.set_xlabel('X label')
            ax.set_ylabel('Y label')
            ax.set_zlabel('Z label')

            plt.xlim(0, 20)
            plt.ylim(10, 30)
            ax.set_zlim(0, 20)


            xs = xS[i]
            ys = yS[i]
            zs = zS[i]
            # s means the size of the dot
            ax.scatter(xs, ys, zs, c=c, s = [(20-5*e) for e in zs], marker=m)


            # 这个plot会连接全部的点
            # ax.plot(xs, ys, zs, c=c)   
            pO = [xs[0], ys[0], zs[0]]

    
            # 肘部原点 O, 从大拇指到小拇指的指尖 依次为A, B, C, D, E
            # 连线中点 分别是 _A1, _B1, _C1, _D1, _E1, 属于容易计算的 直接指尖和肘部原点连线中点 
        
            xsj = [(pO[0] + (e-pO[0])/2) for e in xs[1:]]
            ysj = [(pO[1] + (e-pO[1])/2) for e in ys[1:]]
            zsj = [(pO[2] + (e-pO[2])/2) for e in zs[1:]]
            ax.scatter(xsj, ysj, zsj, c='b', s=1)


            #  initialize the dot pairs for connecting as lines

            dotx, doty, dotz = [0]*2, [0]*2, [0]*2
    

            dotx[0] = xs[0]
            doty[0] = ys[0]
            dotz[0] = zs[0]


            for i in range(1, len(xs)):

                dotx[1] = xs[i]
                doty[1] = ys[i]
                dotz[1] = zs[i]
                ax.plot(dotx, doty, dotz, c='y')

                if i == 2 or i == 5:
                    dotx[1] = xsj[i-1]
                    doty[1] = ysj[i-1]
                    dotz[1] = zsj[i-1]                
                    ax.plot(dotx, doty, dotz, c='g') 


            
            # 食指和小拇指的起点B, E 以及肘部原点的O 构成的三角平面为掌面
            dotx[0] = xsj[1]
            dotx[1] = xsj[4]
            doty[0] = ysj[1]
            doty[1] = ysj[4]
            dotz[0] = zsj[1]
            dotz[1] = zsj[4]
            ax.plot(dotx, doty, dotz, c='g')


            Tx = [pO[0], dotx[0], dotx[1]]
            Ty = [pO[1], doty[0], doty[1]]
            Tz = [pO[2], dotz[0], dotz[1]]
            # 1. create vertices from points
            verts = [list(zip(Tx, Ty, Tz))]
            # 2. create 3d polygons and specify parameters
            srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
            # 3. add polygon to the figure (current axes)
            plt.gca().add_collection3d(srf)




            def euclideanDistance(p1, p2):
                return math.sqrt(((p1[0]-p2[0])**2) + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


            #  求出掌面的方程 点向式方程
            OB= [[xs[0], ys[0], zs[0]], [xS[2], yS[2], zS[2]]]

            p1 = np.array([pO[0], pO[1], pO[2]])
            p2 = np.array([dotx[0], doty[0], dotz[0]])
            p3 = np.array([dotx[1], doty[1], dotz[1]])

            v_p3p1 = p3 - p1
            v_p2p1 = p2 - p1
            
            # 叉积
            cp = np.cross(v_p3p1, v_p2p1)
            # print("CP: ", cp)

            # 平面方程的系数
            A, B, C = cp

            # 点积 
            D = np.dot(cp, p3)

            vNor = [A, B, C]

            # ax.quiver([p1[0]], [p1[1]], [p1[2]], [A], [B], [C], linewidths = (2,), edgecolor = "blue")

            # print("The palm plane is {0}x + {1}y + {2}z = 0")
                


            # 大拇指
            # 假设一节手指长度为0.5， 弯曲角度为 15°， 建立三个方程 找到空间的手指关节点
            F_A_L = 2
            Ang_A0 = 15
            Ang_A1 = 10
            Ang_A2 = 10

            # 大拇指的起点和终点坐标

            pA = [xs[1], ys[1], zs[1]]
            O_A1 = 2*F_A_L*math.cos(math.radians(Ang_A0))
            A1_A2 = F_A_L*math.cos(math.radians(Ang_A1))
            A2_A3 = F_A_L*math.cos(math.radians(Ang_A2))
            len_OA = euclideanDistance(pO, pA)
            O_A1_ratio = O_A1/len_OA
            A1_A2_ratio = A1_A2/len_OA

            

            # print("RATIO, ", O_A1_ratio)

            p_A1 = [pO[0] + O_A1_ratio*(pA[0]-pO[0]), pO[1] + O_A1_ratio*(pA[1]-pO[1]), pO[2] + O_A1_ratio*(pA[2]-pO[2])]
            p_A2 = [pO[0] + (O_A1_ratio+A1_A2_ratio)*(pA[0]-pO[0]), pO[1] + (O_A1_ratio+A1_A2_ratio)*(pA[1]-pO[1]), pO[2] + (O_A1_ratio+A1_A2_ratio)*(pA[2]-pO[2])]
            


            ax.scatter(p_A1[0], p_A1[1], p_A1[2], c='g', marker='o', s = 5)
            ax.scatter(p_A2[0], p_A2[1], p_A2[2], c='g', marker='o', s = 5)

            


         

            # 建立三元方程组 找空中的那一个节点 B1 垂足 _B1 肘部原点O 掌面三角形OBE其他两个点 B 和 E 食指和小指指尖
            
            # 三个关系式
            # 方程1：B1_B1 = F_B_L*sin
            # 方程2：B1_B1 ⊥ OB
            # 方程3：B1_B1 ⊥ OBE
            

            p4_a = np.array([p_A1[0], p_A1[1], p_A1[2]])
            p5_a = np.array([p_A2[0], p_A2[1], p_A2[2]])



            #  x, y, z 是点B1，B2, B3 的坐标
            def sovel_function_A1_A2_A3(unsovled_value):
                x1, y1, z1 = unsovled_value[0], unsovled_value[1], unsovled_value[2]
                pA1 = np.array([x1,y1,z1])
                v3 = pA1 - p4_a
                v4 = p4_a - p1

                x2, y2, z2 = unsovled_value[3], unsovled_value[4], unsovled_value[5]
                pA2 = np.array([x2,y2,z2])
                v5 = pA2 - p5_a
                v6 = p5_a - p1


                return[
                    # 方程1 B1_B1 = L*sin
                    euclideanDistance(pA1, p_A1) - F_A_L*math.sin(math.radians(Ang_A0)),
                    # 方程2 B1_B1 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v3,v4),
                    # 方程3 B1_B1 ⊥ OBE 也就是 O_B1 // vNor  QG 与 平面法向量 平行 点积为0
                    np.inner(v3, vNor),

                    # 方程1 B2_B2 = L*sin
                    euclideanDistance(pA2, p_A2) - F_A_L*math.sin(math.radians(Ang_A1)),
                    # 方程2 B2_B2 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v5,v6),
                    # 方程3 B2_B2 ⊥ OBE 也就是 O_B1 // vNor  B2_B2 与 平面法向量 平行 点积为0
                    np.inner(v5, vNor),

                ]
            
        
            pA1_A2_A3 = fsolve(sovel_function_A1_A2_A3, [0,0,0,0,0,0])

            # 绘制 点 A1, A2
            ax.scatter(pA1_A2_A3[0], pA1_A2_A3[1], pA1_A2_A3[2], c='b', marker='o', s = 10)
            ax.scatter(pA1_A2_A3[3], pA1_A2_A3[4], pA1_A2_A3[5], c='b', marker='o', s = 10)

            # A2 Thumbdistal
            print("A2: %.2f \t" % pA1_A2_A3[3] + "%.2f \t" % pA1_A2_A3[4] + "%.2f \t" % pA1_A2_A3[5])  
            # A1 ThumbProximal
            print("A1: %.2f \t" % pA1_A2_A3[0] + "%.2f \t" % pA1_A2_A3[1] + "%.2f \t" % pA1_A2_A3[2])

            print("\n")



            # 绘制 OA1, A1A2, A2A
            ax.plot([pO[0], pA1_A2_A3[0]], [pO[1], pA1_A2_A3[1]], [pO[2], pA1_A2_A3[2]], c = 'b')
            ax.plot([pA1_A2_A3[0], pA1_A2_A3[3]], [pA1_A2_A3[1], pA1_A2_A3[4]], [pA1_A2_A3[2], pA1_A2_A3[5]], c = 'b')
            ax.plot([pA1_A2_A3[3], pA[0]], [pA1_A2_A3[4], pA[1]], [pA1_A2_A3[5], pA[2]], c = 'b')


            
            # 食指
            # 假设一节手指长度为1.5， 弯曲角度为 15°， 建立三个方程 找到空间的手指关节点
            F_B_L = 2
            Ang_B0 = 15
            Ang_B1 = 10
            Ang_B2 = 10

            # 食指的起点和终点坐标

            pB = [xs[2], ys[2], zs[2]]
            O_B1 = 2*F_B_L*math.cos(math.radians(Ang_B0))
            B1_B2 = F_B_L*math.cos(math.radians(Ang_B1))
            B2_B3 = F_B_L*math.cos(math.radians(Ang_B2))
            len_OB = euclideanDistance(pO, pB)
            O_B1_ratio = O_B1/len_OB
            B1_B2_ratio = B1_B2/len_OB
            B2_B3_ratio = B2_B3/len_OB
            

            # print("RATIO, ", O_B1_ratio)

            p_B1 = [pO[0] + O_B1_ratio*(pB[0]-pO[0]), pO[1] + O_B1_ratio*(pB[1]-pO[1]), pO[2] + O_B1_ratio*(pB[2]-pO[2])]
            p_B2 = [pO[0] + (O_B1_ratio+B1_B2_ratio)*(pB[0]-pO[0]), pO[1] + (O_B1_ratio+B1_B2_ratio)*(pB[1]-pO[1]), pO[2] + (O_B1_ratio+B1_B2_ratio)*(pB[2]-pO[2])]
            p_B3 = [pO[0] + (O_B1_ratio+B1_B2_ratio+B2_B3_ratio)*(pB[0]-pO[0]), pO[1] + (O_B1_ratio+B1_B2_ratio+B2_B3_ratio)*(pB[1]-pO[1]), pO[2] + (O_B1_ratio+B1_B2_ratio+B2_B3_ratio)*(pB[2]-pO[2])]


            ax.scatter(p_B1[0], p_B1[1], p_B1[2], c='g', marker='o', s = 5)
            ax.scatter(p_B2[0], p_B2[1], p_B2[2], c='g', marker='o', s = 5)
            ax.scatter(p_B3[0], p_B3[1], p_B3[2], c='g', marker='o', s = 5)




            # 建立三元方程组 找空中的那一个节点 B1 垂足 _B1 肘部原点O 掌面三角形OBE其他两个点 B 和 E 食指和小指指尖
            
            # 三个关系式
            # 方程1：B1_B1 = F_B_L*sin
            # 方程2：B1_B1 ⊥ OB
            # 方程3：B1_B1 ⊥ OBE
            

            p4 = np.array([p_B1[0], p_B1[1], p_B1[2]])
            p5 = np.array([p_B2[0], p_B2[1], p_B2[2]])
            p6 = np.array([p_B3[0], p_B3[1], p_B3[2]])


            #  x, y, z 是点B1，B2, B3 的坐标
            def sovel_function_B1_B2_B3(unsovled_value):
                x1, y1, z1 = unsovled_value[0], unsovled_value[1], unsovled_value[2]
                pB1 = np.array([x1,y1,z1])
                v3 = pB1 - p4
                v4 = p4 - p1

                x2, y2, z2 = unsovled_value[3], unsovled_value[4], unsovled_value[5]
                pB2 = np.array([x2,y2,z2])
                v5 = pB2 - p5
                v6 = p5 - p1

                x3, y3, z3 = unsovled_value[6], unsovled_value[7], unsovled_value[8]
                pB3 = np.array([x3,y3,z3])
                v7 = pB3 - p6
                v8 = p6 - p1

                return[
                    # 方程1 B1_B1 = L*sin
                    euclideanDistance(pB1, p_B1) - F_B_L*math.sin(math.radians(Ang_B0)),
                    # 方程2 B1_B1 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v3,v4),
                    # 方程3 B1_B1 ⊥ OBE 也就是 O_B1 // vNor  QG 与 平面法向量 平行 点积为0
                    np.inner(v3, vNor),

                    # 方程1 B2_B2 = L*sin
                    euclideanDistance(pB2, p_B2) - F_B_L*math.sin(math.radians(Ang_B1)),
                    # 方程2 B2_B2 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v5,v6),
                    # 方程3 B2_B2 ⊥ OBE 也就是 O_B1 // vNor  B2_B2 与 平面法向量 平行 点积为0
                    np.inner(v5, vNor),

                    # 方程1 B3_B3 = L*sin
                    euclideanDistance(pB3, p_B3) - F_B_L*math.sin(math.radians(Ang_B2)),
                    # 方程2 B3_B3 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v7,v8),
                    # 方程3 B3_B3 ⊥ OBE 也就是 O_B1 // vNor  B3_B3 与 平面法向量 平行 点积为0
                    np.inner(v7, vNor),

                ]
            
        
            pB1_B2_B3 = fsolve(sovel_function_B1_B2_B3, [0,0,0,0,0,0,0,0,0])

            # 绘制 点 B1, B2, B3
            ax.scatter(pB1_B2_B3[0], pB1_B2_B3[1], pB1_B2_B3[2], c='b', marker='o', s = 10)
            ax.scatter(pB1_B2_B3[3], pB1_B2_B3[4], pB1_B2_B3[5], c='b', marker='o', s = 10)
            ax.scatter(pB1_B2_B3[6], pB1_B2_B3[7], pB1_B2_B3[8], c='b', marker='o', s = 10)
            # 绘制 OB1, B1B2, B2B3, B3B
            ax.plot([pO[0], pB1_B2_B3[0]], [pO[1], pB1_B2_B3[1]], [pO[2], pB1_B2_B3[2]], c = 'b')
            ax.plot([pB1_B2_B3[0], pB1_B2_B3[3]], [pB1_B2_B3[1], pB1_B2_B3[4]], [pB1_B2_B3[2], pB1_B2_B3[5]], c = 'b')
            ax.plot([pB1_B2_B3[3], pB1_B2_B3[6]], [pB1_B2_B3[4], pB1_B2_B3[7]], [pB1_B2_B3[5], pB1_B2_B3[8]], c = 'b')
            ax.plot([pB1_B2_B3[6], pB[0]], [pB1_B2_B3[7], pB[1]], [pB1_B2_B3[8], pB[2]], c = 'b')


            # B3 IndexDistal
            print("B3: %.2f \t" % pB1_B2_B3[6] + "%.2f \t" % pB1_B2_B3[7] + "%.2f \t" % pB1_B2_B3[8])
            # B2 IndexMiddle
            print("B2: %.2f \t" % pB1_B2_B3[3] + "%.2f \t" % pB1_B2_B3[4] + "%.2f \t" % pB1_B2_B3[5])
            # B1 IndexKnuckle
            print("B1: %.2f \t" % pB1_B2_B3[0] + "%.2f \t" % pB1_B2_B3[1] + "%.2f \t" % pB1_B2_B3[2])

            print("\n")      

            # 中指
            # 假设一节手指长度为1.5， 弯曲角度为 15°， 建立三个方程 找到空间的手指关节点
            F_C_L = 4
            Ang_C0 = 15
            Ang_C1 = 10
            Ang_C2 = 5

            # 中指的起点和终点坐标

            pC = [xs[3], ys[3], zs[3]]
            O_C1 = 2*F_C_L*math.cos(math.radians(Ang_C0))
            C1_C2 = F_C_L*math.cos(math.radians(Ang_C1))
            C2_C3 = F_C_L*math.cos(math.radians(Ang_C2))
            len_OC = euclideanDistance(pO, pC)
            O_C1_ratio = O_C1/len_OC
            C1_C2_ratio = C1_C2/len_OC
            C2_C3_ratio = C2_C3/len_OC
            

            # print("RATIO, ", O_C1_ratio)

            p_C1 = [pO[0] + O_C1_ratio*(pC[0]-pO[0]), pO[1] + O_C1_ratio*(pC[1]-pO[1]), pO[2] + O_C1_ratio*(pC[2]-pO[2])]
            p_C2 = [pO[0] + (O_C1_ratio+C1_C2_ratio)*(pC[0]-pO[0]), pO[1] + (O_C1_ratio+C1_C2_ratio)*(pC[1]-pO[1]), pO[2] + (O_C1_ratio+C1_C2_ratio)*(pC[2]-pO[2])]
            p_C3 = [pO[0] + (O_C1_ratio+C1_C2_ratio+C2_C3_ratio)*(pC[0]-pO[0]), pO[1] + (O_C1_ratio+C1_C2_ratio+C2_C3_ratio)*(pC[1]-pO[1]), pO[2] + (O_C1_ratio+C1_C2_ratio+C2_C3_ratio)*(pC[2]-pO[2])]


            ax.scatter(p_C1[0], p_C1[1], p_C1[2], c='g', marker='o', s = 5)
            ax.scatter(p_C2[0], p_C2[1], p_C2[2], c='g', marker='o', s = 5)
            ax.scatter(p_C3[0], p_C3[1], p_C3[2], c='g', marker='o', s = 5)


 

            # 建立三元方程组 找空中的那一个节点 C1 垂足 _C1 肘部原点O 掌面三角形OBE其他两个点 B 和 E 食指和小指指尖
            
            # 三个关系式
            # 方程1：C1_C1 = F_C_L*sin
            # 方程2：C1_C1 ⊥ OC
            # 方程3：C1_C1 ⊥ OBE
            

            p4_c = np.array([p_C1[0], p_C1[1], p_C1[2]])
            p5_c = np.array([p_C2[0], p_C2[1], p_C2[2]])
            p6_c = np.array([p_C3[0], p_C3[1], p_C3[2]])


            #  x, y, z 是点C1，C2, C3 的坐标
            def sovel_function_C1_C2_C3(unsovled_value):
                x1, y1, z1 = unsovled_value[0], unsovled_value[1], unsovled_value[2]
                pC1 = np.array([x1,y1,z1])
                v3 = pC1 - p4_c
                v4 = p4_c - p1

                x2, y2, z2 = unsovled_value[3], unsovled_value[4], unsovled_value[5]
                pC2 = np.array([x2,y2,z2])
                v5 = pC2 - p5_c
                v6 = p5_c - p1

                x3, y3, z3 = unsovled_value[6], unsovled_value[7], unsovled_value[8]
                pC3 = np.array([x3,y3,z3])
                v7 = pC3 - p6_c
                v8 = p6_c - p1

                return[
                    # 方程1 C1_C1 = L*sin
                    euclideanDistance(pC1, p_C1) - F_C_L*math.sin(math.radians(Ang_C0)),
                    # 方程2 C1_C1 ⊥ O_C1    也就是这两个向量叉乘为0
                    np.dot(v3,v4),
                    # 方程3 C1_C1 ⊥ OBE 
                    np.inner(v3, vNor),

                    # 方程1 C2_C2 = L*sin
                    euclideanDistance(pC2, p_C2) - F_C_L*math.sin(math.radians(Ang_C1)),
                    # 方程2 C2_C2 ⊥ O_C1    也就是这两个向量叉乘为0
                    np.dot(v5,v6),
                    # 方程3 C2_C2 ⊥ OBE 
                    np.inner(v5, vNor),

                    # 方程1 C3_C3 = L*sin
                    euclideanDistance(pC3, p_C3) - F_C_L*math.sin(math.radians(Ang_C2)),
                    # 方程2 C3_C3 ⊥ O_C1    也就是这两个向量叉乘为0
                    np.dot(v7,v8),
                    # 方程3 C3_C3 ⊥ OBE 也就是 O_B1 // vNor  B3_B3 与 平面法向量 平行 点积为0
                    np.inner(v7, vNor),

                ]
            
        
            pC1_C2_C3 = fsolve(sovel_function_C1_C2_C3, [0,0,0,0,0,0,0,0,0])

            # 绘制 点 C1, C2, C3
            ax.scatter(pC1_C2_C3[0], pC1_C2_C3[1], pC1_C2_C3[2], c='b', marker='o', s = 10)
            ax.scatter(pC1_C2_C3[3], pC1_C2_C3[4], pC1_C2_C3[5], c='b', marker='o', s = 10)
            ax.scatter(pC1_C2_C3[6], pC1_C2_C3[7], pC1_C2_C3[8], c='b', marker='o', s = 10)
            # 绘制 OB1, B1B2, B2B3, B3B
            ax.plot([pO[0], pC1_C2_C3[0]], [pO[1], pC1_C2_C3[1]], [pO[2], pC1_C2_C3[2]], c = 'b')
            ax.plot([pC1_C2_C3[0], pC1_C2_C3[3]], [pC1_C2_C3[1], pC1_C2_C3[4]], [pC1_C2_C3[2], pC1_C2_C3[5]], c = 'b')
            ax.plot([pC1_C2_C3[3], pC1_C2_C3[6]], [pC1_C2_C3[4], pC1_C2_C3[7]], [pC1_C2_C3[5], pC1_C2_C3[8]], c = 'b')
            ax.plot([pC1_C2_C3[6], pC[0]], [pC1_C2_C3[7], pC[1]], [pC1_C2_C3[8], pC[2]], c = 'b')

            # C3 MiddleDistal
            print("C3: %.2f \t" % pC1_C2_C3[6] + "%.2f \t" % pC1_C2_C3[7] + "%.2f \t" % pC1_C2_C3[8])
            # C2 MiddleMiddle
            print("C2: %.2f \t" % pC1_C2_C3[3] + "%.2f \t" % pC1_C2_C3[4] + "%.2f \t" % pC1_C2_C3[5])
            # C1 MiddleKnuckle
            print("C1: %.2f \t" % pC1_C2_C3[0] + "%.2f \t" % pC1_C2_C3[1] + "%.2f \t" % pC1_C2_C3[2])
            
            print("\n")     


            # 无名指
            # 假设一节手指长度为1.5， 弯曲角度为 15°， 建立三个方程 找到空间的手指关节点
            F_D_L = 2
            Ang_D0 = 15
            Ang_D1 = 15
            Ang_D2 = 15

            # 无名指的起点和终点坐标

            pD = [xs[4], ys[4], zs[4]]
            O_D1 = 2*F_D_L*math.cos(math.radians(Ang_D0))
            D1_D2 = F_D_L*math.cos(math.radians(Ang_D1))
            D2_D3 = F_D_L*math.cos(math.radians(Ang_D2))
            len_OD = euclideanDistance(pO, pD)
            O_D1_ratio = O_D1/len_OD
            D1_D2_ratio = D1_D2/len_OD
            D2_D3_ratio = D2_D3/len_OD
            

            # print("RATIO, ", O_D1_ratio)

            p_D1 = [pO[0] + O_D1_ratio*(pD[0]-pO[0]), pO[1] + O_D1_ratio*(pD[1]-pO[1]), pO[2] + O_D1_ratio*(pD[2]-pO[2])]
            p_D2 = [pO[0] + (O_D1_ratio+D1_D2_ratio)*(pD[0]-pO[0]), pO[1] + (O_D1_ratio+D1_D2_ratio)*(pD[1]-pO[1]), pO[2] + (O_D1_ratio+D1_D2_ratio)*(pD[2]-pO[2])]
            p_D3 = [pO[0] + (O_D1_ratio+D1_D2_ratio+D2_D3_ratio)*(pD[0]-pO[0]), pO[1] + (O_D1_ratio+D1_D2_ratio+D2_D3_ratio)*(pD[1]-pO[1]), pO[2] + (O_D1_ratio+D1_D2_ratio+D2_D3_ratio)*(pD[2]-pO[2])]


            ax.scatter(p_D1[0], p_D1[1], p_D1[2], c='g', marker='o', s = 5)
            ax.scatter(p_D2[0], p_D2[1], p_D2[2], c='g', marker='o', s = 5)
            ax.scatter(p_D3[0], p_D3[1], p_D3[2], c='g', marker='o', s = 5)



            # 建立三元方程组 找空中的那一个节点 B1 垂足 _B1 肘部原点O 掌面三角形OBE其他两个点 B 和 E 食指和小指指尖
            
            # 三个关系式
            # 方程1：B1_B1 = F_B_L*sin
            # 方程2：B1_B1 ⊥ OB
            # 方程3：B1_B1 ⊥ OBE
            

            p4_d = np.array([p_D1[0], p_D1[1], p_D1[2]])
            p5_d = np.array([p_D2[0], p_D2[1], p_D2[2]])
            p6_d = np.array([p_D3[0], p_D3[1], p_D3[2]])


            #  x, y, z 是点B1，B2, B3 的坐标
            def sovel_function_D1_D2_D3(unsovled_value):
                x1, y1, z1 = unsovled_value[0], unsovled_value[1], unsovled_value[2]
                pD1 = np.array([x1,y1,z1])
                v3 = pD1 - p4_d
                v4 = p4_d - p1

                x2, y2, z2 = unsovled_value[3], unsovled_value[4], unsovled_value[5]
                pD2 = np.array([x2,y2,z2])
                v5 = pD2 - p5_d
                v6 = p5_d - p1

                x3, y3, z3 = unsovled_value[6], unsovled_value[7], unsovled_value[8]
                pD3 = np.array([x3,y3,z3])
                v7 = pD3 - p6_d
                v8 = p6_d - p1

                return[
                    # 方程1 B1_B1 = L*sin
                    euclideanDistance(pD1, p_D1) - F_D_L*math.sin(math.radians(Ang_D0)),
                    # 方程2 B1_B1 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v3,v4),
                    # 方程3 B1_B1 ⊥ OBE 也就是 O_B1 // vNor  QG 与 平面法向量 平行 点积为0
                    np.inner(v3, vNor),

                    # 方程1 B2_B2 = L*sin
                    euclideanDistance(pD2, p_D2) - F_D_L*math.sin(math.radians(Ang_D1)),
                    # 方程2 B2_B2 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v5,v6),
                    # 方程3 B2_B2 ⊥ OBE 也就是 O_B1 // vNor  B2_B2 与 平面法向量 平行 点积为0
                    np.inner(v5, vNor),

                    # 方程1 B3_B3 = L*sin
                    euclideanDistance(pD3, p_D3) - F_D_L*math.sin(math.radians(Ang_D2)),
                    # 方程2 B3_B3 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v7,v8),
                    # 方程3 B3_B3 ⊥ OBE 也就是 O_B1 // vNor  B3_B3 与 平面法向量 平行 点积为0
                    np.inner(v7, vNor),

                ]
            
        
            pD1_D2_D3 = fsolve(sovel_function_D1_D2_D3, [0,0,0,0,0,0,0,0,0])

            # 绘制 点 B1, B2, B3
            ax.scatter(pD1_D2_D3[0], pD1_D2_D3[1], pD1_D2_D3[2], c='b', marker='o', s = 10)
            ax.scatter(pD1_D2_D3[3], pD1_D2_D3[4], pD1_D2_D3[5], c='b', marker='o', s = 10)
            ax.scatter(pD1_D2_D3[6], pD1_D2_D3[7], pD1_D2_D3[8], c='b', marker='o', s = 10)
            # 绘制 OB1, B1B2, B2B3, B3B
            ax.plot([pO[0], pD1_D2_D3[0]], [pO[1], pD1_D2_D3[1]], [pO[2], pD1_D2_D3[2]], c = 'b')
            ax.plot([pD1_D2_D3[0], pD1_D2_D3[3]], [pD1_D2_D3[1], pD1_D2_D3[4]], [pD1_D2_D3[2], pD1_D2_D3[5]], c = 'b')
            ax.plot([pD1_D2_D3[3], pD1_D2_D3[6]], [pD1_D2_D3[4], pD1_D2_D3[7]], [pD1_D2_D3[5], pD1_D2_D3[8]], c = 'b')
            ax.plot([pD1_D2_D3[6], pD[0]], [pD1_D2_D3[7], pD[1]], [pD1_D2_D3[8], pD[2]], c = 'b')

            # D3  RingDistal
            print("D3: %.2f \t" % pD1_D2_D3[6] + "%.2f \t" % pD1_D2_D3[7] + "%.2f \t" % pD1_D2_D3[8])  
            # D2  RingMiddle
            print("D2: %.2f \t" % pD1_D2_D3[3] + "%.2f \t" % pD1_D2_D3[4] + "%.2f \t" % pD1_D2_D3[5])
            # D1 RingKnuckle
            print("D1: %.2f \t" % pD1_D2_D3[0] + "%.2f \t" % pD1_D2_D3[1] + "%.2f \t" % pD1_D2_D3[2])

            print("\n")


            # 小指
            # 假设一节手指长度为1， 弯曲角度为 15°， 建立三个方程 找到空间的手指关节点
            F_E_L = 2
            Ang_E0 = 15
            Ang_E1 = 15
            Ang_E2 = 15

            # 食指的起点和终点坐标

            pE = [xs[5], ys[5], zs[5]]
            O_E1 = 2*F_E_L*math.cos(math.radians(Ang_E0))
            E1_E2 = F_E_L*math.cos(math.radians(Ang_E1))
            E2_E3 = F_E_L*math.cos(math.radians(Ang_E2))
            len_OE = euclideanDistance(pO, pE)
            O_E1_ratio = O_E1/len_OE
            E1_E2_ratio = E1_E2/len_OE
            E2_E3_ratio = E2_E3/len_OE
            

            # print("RATIO, ", O_E1_ratio)

            p_E1 = [pO[0] + O_E1_ratio*(pE[0]-pO[0]), pO[1] + O_E1_ratio*(pE[1]-pO[1]), pO[2] + O_E1_ratio*(pE[2]-pO[2])]
            p_E2 = [pO[0] + (O_E1_ratio+E1_E2_ratio)*(pE[0]-pO[0]), pO[1] + (O_E1_ratio+E1_E2_ratio)*(pE[1]-pO[1]), pO[2] + (O_E1_ratio+E1_E2_ratio)*(pE[2]-pO[2])]
            p_E3 = [pO[0] + (O_E1_ratio+E1_E2_ratio+E2_E3_ratio)*(pE[0]-pO[0]), pO[1] + (O_E1_ratio+E1_E2_ratio+E2_E3_ratio)*(pE[1]-pO[1]), pO[2] + (O_E1_ratio+E1_E2_ratio+E2_E3_ratio)*(pE[2]-pO[2])]


            ax.scatter(p_E1[0], p_E1[1], p_E1[2], c='g', marker='o', s = 5)
            ax.scatter(p_E2[0], p_E2[1], p_E2[2], c='g', marker='o', s = 5)
            ax.scatter(p_E3[0], p_E3[1], p_E3[2], c='g', marker='o', s = 5)




            # 建立三元方程组 找空中的那一个节点 B1 垂足 _B1 肘部原点O 掌面三角形OBE其他两个点 B 和 E 食指和小指指尖
            
            # 三个关系式
            # 方程1：B1_B1 = F_B_L*sin
            # 方程2：B1_B1 ⊥ OB
            # 方程3：B1_B1 ⊥ OBE
            

            p4_e = np.array([p_E1[0], p_E1[1], p_E1[2]])
            p5_e = np.array([p_E2[0], p_E2[1], p_E2[2]])
            p6_e = np.array([p_E3[0], p_E3[1], p_E3[2]])


            #  x, y, z 是点B1，B2, B3 的坐标
            def sovel_function_E1_E2_E3(unsovled_value):
                x1, y1, z1 = unsovled_value[0], unsovled_value[1], unsovled_value[2]
                pE1 = np.array([x1,y1,z1])
                v3 = pE1 - p4_e
                v4 = p4_e - p1

                x2, y2, z2 = unsovled_value[3], unsovled_value[4], unsovled_value[5]
                pE2 = np.array([x2,y2,z2])
                v5 = pE2 - p5_e
                v6 = p5_e - p1

                x3, y3, z3 = unsovled_value[6], unsovled_value[7], unsovled_value[8]
                pE3 = np.array([x3,y3,z3])
                v7 = pE3 - p6_e
                v8 = p6_e - p1

                return[
                    # 方程1 B1_B1 = L*sin
                    euclideanDistance(pE1, p_E1) - F_E_L*math.sin(math.radians(Ang_E0)),
                    # 方程2 B1_B1 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v3,v4),
                    # 方程3 B1_B1 ⊥ OBE 也就是 O_B1 // vNor  QG 与 平面法向量 平行 点积为0
                    np.inner(v3, vNor),

                    # 方程1 B2_B2 = L*sin
                    euclideanDistance(pE2, p_E2) - F_E_L*math.sin(math.radians(Ang_E1)),
                    # 方程2 B2_B2 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v5,v6),
                    # 方程3 B2_B2 ⊥ OBE 也就是 O_B1 // vNor  B2_B2 与 平面法向量 平行 点积为0
                    np.inner(v5, vNor),

                    # 方程1 B3_B3 = L*sin
                    euclideanDistance(pE3, p_E3) - F_E_L*math.sin(math.radians(Ang_E2)),
                    # 方程2 B3_B3 ⊥ O_B1    也就是这两个向量叉乘为0
                    np.dot(v7,v8),
                    # 方程3 B3_B3 ⊥ OBE 也就是 O_B1 // vNor  B3_B3 与 平面法向量 平行 点积为0
                    np.inner(v7, vNor),

                ]
            
        
            pE1_E2_E3 = fsolve(sovel_function_E1_E2_E3, [0,0,0,0,0,0,0,0,0])

            # 绘制 点 B1, B2, B3
            ax.scatter(pE1_E2_E3[0], pE1_E2_E3[1], pE1_E2_E3[2], c='b', marker='o', s = 10)
            ax.scatter(pE1_E2_E3[3], pE1_E2_E3[4], pE1_E2_E3[5], c='b', marker='o', s = 10)
            ax.scatter(pE1_E2_E3[6], pE1_E2_E3[7], pE1_E2_E3[8], c='b', marker='o', s = 10)
            # 绘制 OB1, B1B2, B2B3, B3B
            ax.plot([pO[0], pE1_E2_E3[0]], [pO[1], pE1_E2_E3[1]], [pO[2], pE1_E2_E3[2]], c = 'b')
            ax.plot([pE1_E2_E3[0], pE1_E2_E3[3]], [pE1_E2_E3[1], pE1_E2_E3[4]], [pE1_E2_E3[2], pE1_E2_E3[5]], c = 'b')
            ax.plot([pE1_E2_E3[3], pE1_E2_E3[6]], [pE1_E2_E3[4], pE1_E2_E3[7]], [pE1_E2_E3[5], pE1_E2_E3[8]], c = 'b')
            ax.plot([pE1_E2_E3[6], pE[0]], [pE1_E2_E3[7], pE[1]], [pE1_E2_E3[8], pE[2]], c = 'b')

            # E3 littleDistal
            print("E3: %.2f \t" % pE1_E2_E3[6] + "%.2f \t" % pE1_E2_E3[7] + "%.2f \t" % pE1_E2_E3[8])
            # E2 littleMiddle
            print("E2: %.2f \t" % pE1_E2_E3[3] + "%.2f \t" % pE1_E2_E3[4] + "%.2f \t" % pE1_E2_E3[5])
            # E1 littleKnucle
            print("E1: %.2f \t" % pE1_E2_E3[0] + "%.2f \t" % pE1_E2_E3[1] + "%.2f \t" % pE1_E2_E3[2])

            print("\n")
            print("\n")

            # dynamicly update the figures
            fig.canvas.draw()
            fig.canvas.flush_events()

            end = time.time()
            # print ("time: ", end-start)

            print("\n")

            time.sleep(0.2)
        



        
        





