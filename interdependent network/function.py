import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

nodenumber = 50
average = 1
graph1 = nx.barabasi_albert_graph(nodenumber, average)
graph2 = nx.barabasi_albert_graph(nodenumber, average)
ps1 = nx.spring_layout(graph1)
ps2 = nx.random_layout(graph2)
nx.draw(graph1, ps1, with_labels=False, node_size=nodenumber)
plt.title("network1")
nx.draw(graph2, ps2, with_labels=False, node_size=nodenumber)
plt.title("network2")
G = nx.union(graph1, graph2, rename=('Gone', 'Gtwo'))
psG = nx.spring_layout(G)
nx.draw(G, psG, with_labels=False, node_size=30)
# 相依边添加，可定义相依边edge=[]，G.add_weighted_edges_from(edges)
match = 2
if match == 2:  # 正序匹配

    degree_1 = [0] * nodenumber

    for i in range(0, nodenumber):
        degree_1[i] = graph1.degree(i)

    degree_2 = [0] * nodenumber

    for i in range(0, nodenumber):
        degree_2[i] = graph2.degree(i)

    rankdegree_1 = np.argsort(degree_1)

    rankdegree_2 = np.argsort(degree_2)
tempGone = 'Gone{ione}'
tempGtwo = 'Gtwo{itwo}'
for i in range(0, 50):
    G.add_edge(tempGone.format(ione=rankdegree_1[i]), tempGtwo.format(itwo=rankdegree_2[i]))

    G[tempGone.format(ione=rankdegree_1[i])][tempGtwo.format(itwo=rankdegree_2[i])]['lineweight'] = 2  # 网络之间的为虚线

for i in range(0, 50):

    for j in range(0, 50):

        if G.has_edge(tempGone.format(ione=rankdegree_1[i]), tempGone.format(ione=rankdegree_1[j])):
            G[tempGone.format(ione=rankdegree_1[i])][tempGone.format(ione=rankdegree_1[j])]['lineweight'] = 1  # 层内的为实线

for i in range(0, 50):

    for j in range(0, 50):

        if G.has_edge(tempGtwo.format(itwo=rankdegree_1[i]), tempGtwo.format(itwo=rankdegree_1[j])):
            G[tempGtwo.format(itwo=rankdegree_1[i])][tempGtwo.format(itwo=rankdegree_1[j])]['lineweight'] = 1
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['lineweight'] == 1]

esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['lineweight'] == 2]
nx.draw_networkx_nodes(G, psG, node_size=70)

nx.draw_networkx_edges(G, psG, edgelist=elarge, width=1)

nx.draw_networkx_edges(G, psG, edgelist=esmall, width=1, alpha=0.21, edge_color='b', style='dashed')

plt.title("union network")  # 组合图
plt.show()  # 展示相依图

'''
级联失效过程
'''


def fault(graph, node):
    nodes = [(u, v) for (u, v, d) in G.edges(data=True) if d['lineweight'] == 2]
    listn = []
    for j in graph.nodes():
        listn.append(j)
    count1 = len(listn)
    m = 0
    while m < len(nodes):
        if nodes[m][0] == node or nodes[m][1] == node:
            graph.remove_node(nodes[m][0])
            graph.remove_node(nodes[m][1])
            index1 = listn.index(nodes[m][0])
            listn.pop(index1)
            index2 = listn.index(nodes[m][1])
            listn.pop(index2)
            nodes.pop(m)
        else:
            m = m + 1
    if len(graph.nodes()) == count1:
        graph.remove_node(node)
    count2 = len(listn)
    while True:
        if count1 == count2:
            break
        else:
            count1 = len(listn)
            n = 0
            while n < len(listn):
                if graph.degree(listn[n]) == 0:
                    graph.remove_node(listn[n])
                    listn.pop(n)
                elif graph.degree(listn[n]) == 1:
                    q = 0
                    nodem = listn[n]
                    while q < len(nodes):
                        if nodes[q][0] == nodem or nodes[q][1] == nodem:
                            graph.remove_node(nodes[q][0])
                            graph.remove_node(nodes[q][1])
                            index3 = listn.index(nodes[q][0])
                            listn.pop(index3)
                            index4 = listn.index(nodes[q][1])
                            listn.pop(index4)
                            nodes.pop(q)
                            break
                        else:
                            q = q + 1
                else:
                    n = n + 1
            count2 = len(listn)


'''
判断稳定状态下的相依网络存在的节点个数
相依网络G，自己所属的网络为Gone
'''


def fun(graph, net):
    n = 0
    for i in graph.nodes():
        if i.startswith(net):
            n = n + 1
    return n


print(fun(G, "Gone"))
fault(G, "Gone1")
print(fun(G, "Gone"))
