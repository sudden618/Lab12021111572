import networkx as nx
import matplotlib.pyplot as plt
import re
import random
from tkinter import Tk, filedialog


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return words


def build_directed_graph(words):
    graph = nx.DiGraph()
    word_pairs = zip(words, words[1:])
    for a, b in word_pairs:
        if graph.has_edge(a, b):
            graph[a][b]['weight'] += 1
        else:
            graph.add_edge(a, b, weight=1)
    return graph

def draw_graph(graph, shortest_paths=None):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 8))

    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue',
            font_size=15, font_weight='bold', edge_color='gray')

    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    if shortest_paths:
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        for i, path in enumerate(shortest_paths):
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=colors[i % len(colors)], width=2)
            nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color=colors[i % len(colors)])

    plt.title('Directed Graph of Text')
    plt.show()


def save_graph(graph, file_path, shortest_paths=None):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue',
            font_size=15, font_weight='bold', edge_color='gray')
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    if shortest_paths:
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        for i, path in enumerate(shortest_paths):
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=colors[i % len(colors)], width=2)
            nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color=colors[i % len(colors)])

    plt.title('Directed Graph of Text')
    plt.savefig(file_path)
    plt.close()


def find_bridge_words(graph, word1, word2):
    if word1 not in graph.nodes:
        return f"No {word1} in the graph!"
    if word2 not in graph.nodes:
        return f"No {word2} in the graph!"

    bridge_words = []
    for neighbor in graph.successors(word1):
        if graph.has_edge(neighbor, word2):
            bridge_words.append(neighbor)

    if not bridge_words:
        return f"No bridge words from {word1} to {word2}!"
    else:
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
        if word1 in graph and word2 in graph:
            for neighbor in graph.successors(word1):
                if graph.has_edge(neighbor, word2):
                    bridge_words.append(neighbor)

        if bridge_words:
            new_text.append(random.choice(bridge_words))

    new_text.append(words[-1])
    return ' '.join(new_text)


def find_shortest_paths(graph, word1, word2):
    if word1 not in graph.nodes or word2 not in graph.nodes:
        return None, f"No path between {word1} and {word2} because one or both are not in the graph!"

    try:
        all_paths = list(nx.all_shortest_paths(graph, source=word1, target=word2, weight='weight'))
        path_length = nx.shortest_path_length(graph, source=word1, target=word2, weight='weight')
        return all_paths, f"All shortest paths from {word1} to {word2} have length {path_length}."
    except nx.NetworkXNoPath:
        return None, f"No path between {word1} and {word2}!"


def find_single_source_shortest_paths(graph, word):
    if word not in graph.nodes:
        return None, f"No paths from {word} because it is not in the graph!"

    paths = nx.single_source_dijkstra_path(graph, source=word, weight='weight')
    path_lengths = nx.single_source_dijkstra_path_length(graph, source=word, weight='weight')

    return paths, path_lengths


def random_walk(graph):
    current_node = random.choice(list(graph.nodes))
    visited_edges = set()
    path = [current_node]

    print(f"Starting random walk from: {current_node}")

    while True:
        neighbors = list(graph.successors(current_node))
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

    save_path = r"D:\pythonProject\software\lab1\output.png"
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

    #随机游走
    walk_path = random_walk(graph)
    random_walk_path_file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if random_walk_path_file:
        save_random_walk(walk_path, random_walk_path_file)


if __name__ == "__main__":
    main()
