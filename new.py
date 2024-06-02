import re
import random
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import numpy as np

class DirectedGraph:
    def __init__(self):
        self.edges = {}  # 存储图中的边，以字典形式表示邻接表
        self.weights = {}  # 存储每条边的权重，以字典形式表示

    def add_edge(self, from_node, to_node, weight=1):
        # 添加一条从 from_node 到 to_node 的有向边，可以指定权重，默认为1
        if from_node in self.edges:
            if to_node in self.edges[from_node]:  # 如果该边已经存在，则增加权重
                self.weights[(from_node, to_node)] += weight
            else:
                self.edges[from_node].append(to_node)  # 如果该边不存在，则添加新的边
                self.weights[(from_node, to_node)] = weight
        else:
            self.edges[from_node] = [to_node]  # 如果起始节点不存在，则创建起始节点并添加边
            self.weights[(from_node, to_node)] = weight

    def has_edge(self, from_node, to_node):
        # 检查是否存在从 from_node 到 to_node 的边
        return (from_node, to_node) in self.weights

    def successors(self, node):
        # 获取给定节点的所有后继节点
        return self.edges.get(node, [])

    def nodes(self):
        # 获取图中的所有节点
        return set(self.edges.keys()).union(set(v for values in self.edges.values() for v in values))

    def get_edge_weight(self, from_node, to_node):
        # 获取从 from_node 到 to_node 的边的权重
        return self.weights.get((from_node, to_node), None)



def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return words


def build_directed_graph(words):
    # 创建一个 DirectedGraph 实例来构建有向图
    graph = DirectedGraph()
    # 使用 zip 函数将相邻的单词配对，形成单词对
    word_pairs = zip(words, words[1:])
    # 遍历单词对，并将它们作为有向图的边添加到图中
    for a, b in word_pairs:
        graph.add_edge(a, b, weight=1)  # 添加从单词 a 到单词 b 的边，初始权重为 1
    return graph  # 返回构建好的有向图实例

def draw_graph(graph, shortest_paths=None):
    nodes = list(graph.nodes())  # 获取图中所有节点
    pos = {node: (random.random(), random.random()) for node in nodes}  # 为每个节点随机生成位置坐标

    fig, ax = plt.subplots()  # 创建图形和子图对象
    for node, (x, y) in pos.items():  # 遍历节点及其位置信息
        # 在图上绘制节点标签
        ax.text(x, y, node, fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='skyblue', alpha=0.5, edgecolor='black'))

    for (from_node, to_node), weight in graph.weights.items():  # 遍历图中的边和对应的权重
        x_from, y_from = pos[from_node]  # 起始节点的位置坐标
        x_to, y_to = pos[to_node]  # 终止节点的位置坐标
        # 在图上绘制带箭头的边，并标注边的权重
        ax.annotate("",
                    xy=(x_to, y_to), xycoords='data',
                    xytext=(x_from, y_from), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3", color='gray'))
        ax.text((x_from + x_to) / 2, (y_from + y_to) / 2, str(weight),
                fontsize=8, ha='center', va='center', color='red')

    if shortest_paths:  # 如果提供了最短路径信息
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # 路径可视化时使用的颜色列表
        for i, path in enumerate(shortest_paths):  # 遍历最短路径列表
            for j in range(len(path) - 1):  # 遍历路径中的节点
                x_from, y_from = pos[path[j]]  # 当前节点的位置坐标
                x_to, y_to = pos[path[j + 1]]  # 下一个节点的位置坐标
                # 在图上绘制路径
                ax.plot([x_from, x_to], [y_from, y_to], color=colors[i % len(colors)], linewidth=2)

    plt.title('Directed Graph of Text')  # 设置图标题
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图形


def save_graph(graph, file_path, shortest_paths=None):
    nodes = list(graph.nodes())
    pos = {node: (random.random(), random.random()) for node in nodes}

    fig, ax = plt.subplots()
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='skyblue', alpha=0.5, edgecolor='black'))

    for (from_node, to_node), weight in graph.weights.items():
        x_from, y_from = pos[from_node]
        x_to, y_to = pos[to_node]
        ax.annotate("",
                    xy=(x_to, y_to), xycoords='data',
                    xytext=(x_from, y_from), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3", color='gray'))
        ax.text((x_from + x_to) / 2, (y_from + y_to) / 2, str(weight),
                fontsize=8, ha='center', va='center', color='red')

    if shortest_paths:
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        for i, path in enumerate(shortest_paths):
            for j in range(len(path) - 1):
                x_from, y_from = pos[path[j]]
                x_to, y_to = pos[path[j + 1]]
                ax.plot([x_from, x_to], [y_from, y_to], color=colors[i % len(colors)], linewidth=2)

    plt.title('Directed Graph of Text')
    plt.axis('off')
    plt.savefig(file_path)
    plt.close()

def find_bridge_words(graph, word1, word2):
    # 检查第一个单词是否存在于图中
    if word1 not in graph.nodes():
        return f"No {word1} in the graph!"
    # 检查第二个单词是否存在于图中
    if word2 not in graph.nodes():
        return f"No {word2} in the graph!"

    bridge_words = []  # 存储桥接词的列表
    # 遍历第一个单词的后继节点
    for neighbor in graph.successors(word1):
        # 如果存在从当前邻居节点到第二个单词的边，则将邻居节点添加到桥接词列表中
        if graph.has_edge(neighbor, word2):
            bridge_words.append(neighbor)

    if not bridge_words:
        # 如果不存在桥接词，则返回相应的消息
        return f"No bridge words from {word1} to {word2}!"
    else:
        # 如果存在桥接词，则返回桥接词列表的消息
        return f"The bridge words from {word1} to {word2} are: {', '.join(bridge_words)}."


def generate_new_text(graph, text):
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
    words = text.split()
    new_text = []

    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        new_text.append(word1)

        bridge_words = []
        if word1 in graph.nodes() and word2 in graph.nodes():
            for neighbor in graph.successors(word1):
                if graph.has_edge(neighbor, word2):
                    bridge_words.append(neighbor)

        if bridge_words:
            new_text.append(random.choice(bridge_words))

    new_text.append(words[-1])
    return ' '.join(new_text)


def find_shortest_paths(graph, word1, word2):
    if word1 not in graph.nodes() or word2 not in graph.nodes():
        # 如果单词1或单词2不在图中，则返回无法找到路径的消息
        return None, f"由于其中一个或两个单词不在图中，无法在 {word1} 和 {word2} 之间找到路径！"

    # 实现 Dijkstra 算法或任何最短路径算法
    try:
        # 使用 Dijkstra 算法找到所有最短路径
        all_paths = dijkstra_all_shortest_paths(graph, word1, word2)
        # 获取最短路径的长度（第一个路径即为最短路径）
        path_length = len(all_paths[0]) - 1
        # 返回所有最短路径和相应的消息
        return all_paths, f"从 {word1} 到 {word2} 的所有最短路径的长度为 {path_length}。"
    except ValueError:
        # 如果没有找到路径，则返回相应的消息
        return None, f"在 {word1} 和 {word2} 之间找不到路径！"


def dijkstra_all_shortest_paths(graph, start, end):
    import heapq  # 导入 heapq 库，用于实现优先队列
    queue = [(0, start, [])]  # 使用优先队列存储节点、路径的开销和路径
    seen = set()  # 用于记录已经访问过的节点
    min_length = None  # 最短路径的长度
    paths = []  # 存储所有最短路径

    while queue:
        (cost, node, path) = heapq.heappop(queue)  # 从优先队列中弹出具有最小开销的节点和路径
        if node in seen:
            continue

        path = path + [node]  # 将当前节点添加到路径中
        seen.add(node)  # 将当前节点标记为已访问

        if node == end:  # 如果当前节点是终点节点
            if min_length is None:
                min_length = cost
            if cost == min_length:  # 如果当前路径的开销等于最小开销，则将路径添加到最短路径列表中
                paths.append(path)
            continue

        # 遍历当前节点的邻居节点
        for neighbor in graph.successors(node):
            if neighbor not in seen:
                # 获取当前节点到邻居节点的边的权重
                weight = graph.get_edge_weight(node, neighbor)
                # 将邻居节点、路径的总开销和路径添加到优先队列中
                heapq.heappush(queue, (cost + weight, neighbor, path))

    if not paths:
        # 如果最短路径列表为空，则抛出 ValueError 异常
        raise ValueError("未找到路径")
    return paths


def find_single_source_shortest_paths(graph, word):
    if word not in graph.nodes():
        return None, f"因为 {word} 不在图中，所以不存在从 {word} 出发的路径！"

    paths = {}  # 存储从给定单词出发的所有最短路径
    path_lengths = {}  # 存储每个目标单词的最短路径长度
    for target in graph.nodes():
        if target != word:
            try:
                # 使用 Dijkstra 算法找到从给定单词到目标单词的最短路径
                path = dijkstra_all_shortest_paths(graph, word, target)[0]
                paths[target] = path
                # 计算最短路径的长度并存储
                path_lengths[target] = len(path) - 1
            except ValueError:
                continue

    return paths, path_lengths


def random_walk(graph):
    current_node = random.choice(list(graph.nodes()))
    visited_edges = set()
    path = [current_node]

    print(f"Starting random walk from: {current_node}")

    while True:
        neighbors = graph.successors(current_node)
        if not neighbors:
            print(f"Node {current_node} has no outgoing edges. Stopping walk.")
            break

        next_node = random.choice(neighbors)
        edge = (current_node, next_node)

        if edge in visited_edges:
            print(f"Encountered repeated edge {edge}. Stopping walk.")
            break

        visited_edges.add(edge)
        path.append(next_node)
        current_node = next_node

        print(f"Traversed edge: {edge}")

        user_input = input("Press Enter to continue, 's' to stop: ")
        if user_input.lower() == 's':
            print("User stopped the walk.")
            break
    return path

def save_random_walk(path, file_path):
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(' -> '.join(path))
            print(f"Random walk saved to {file_path}")

def main():
            root = Tk()
            root.withdraw()
            file_path = r"D:\pythonProject\software\lab1\test.txt"

            if not file_path:
                print("No file selected.")
                return

            words = read_text_file(file_path)
            graph = build_directed_graph(words)
            draw_graph(graph)

            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                save_graph(graph, save_path)
                print(f"Graph saved to {save_path}")

            # 输入两个单词查询桥接词
            word1 = input("Enter the first word: ").lower()
            word2 = input("Enter the second word: ").lower()
            result = find_bridge_words(graph, word1, word2)
            print(result)

            # 输入一行新文本，生成包含桥接词的新文本
            new_text_input = input("Enter a line of new text: ")
            new_text = generate_new_text(graph, new_text_input)
            print(f"Generated text: {new_text}")

            # 输入两个单词查询所有最短路径
            word1 = input("Enter the first word for shortest paths: ").lower()
            word2 = input("Enter the second word for shortest paths: ").lower()
            all_shortest_paths, path_result = find_shortest_paths(graph, word1, word2)
            if all_shortest_paths:
                draw_graph(graph, all_shortest_paths)
                save_graph(graph, save_path, all_shortest_paths)
            print(path_result)

            # 输入一个单词查询到其他单词的最短路径
            single_word = input("Enter a single word for shortest paths to all other nodes: ").lower()
            single_source_paths, single_source_lengths = find_single_source_shortest_paths(graph, single_word)
            if single_source_paths:
                for target_word, path in single_source_paths.items():
                    path_length = single_source_lengths[target_word]
                    print(
                        f"The shortest path from {single_word} to {target_word} is {' -> '.join(path)} with length {path_length}.")
                    draw_graph(graph, [path])
                    save_graph(graph, save_path, [path])

            # 随机游走
            walk_path = random_walk(graph)
            random_walk_path_file = filedialog.asksaveasfilename(defaultextension=".txt",
                                                                 filetypes=[("Text files", "*.txt")])
            if random_walk_path_file:
                save_random_walk(walk_path, random_walk_path_file)

if __name__ == "__main__":
            main()
