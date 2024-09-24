import os
import time
import pandas as pd

class MaxHeap:
    # Constructs a max heap that consists of nodes which include [delta, node_id].
    # self.indices maintains the indices of the nodes so the delta values can be altered.
    def __init__(self, nodes=[]):
        self.heap = []
        self.indices = {}
        for i in nodes:
            self.add_to_heap(i)
    
    def add_to_heap(self, node):
        assert len(node) == 2
        self.heap.append(node)
        end_index = len(self.heap)-1
        self.indices[node[1]] = end_index
        self.heapify_up(end_index)
    
    def heapify_up(self, ind):
        if ind == 0:
            return
        parent = (ind-1)//2
        if self.heap[parent] >= self.heap[ind]:
            return
        temp = self.heap[ind]
        self.heap[ind] = self.heap[parent]
        self.heap[parent] = temp

        node = self.heap[ind][1]
        self.indices[node] = ind
        parent_node = self.heap[parent][1]
        self.indices[parent_node] = parent

        self.heapify_up(parent)

    def heapify_down(self, ind):
        right = (ind+1)*2
        left = right-1
        if left >= len(self.heap):
            return
        if left == len(self.heap)-1:
            next_ind = left
        else:
            next_ind = right if self.heap[right] > self.heap[left] else left
        if self.heap[next_ind] <= self.heap[ind]:
            return
        temp = self.heap[ind]
        self.heap[ind] = self.heap[next_ind]
        self.heap[next_ind] = temp

        node = self.heap[ind][1]
        self.indices[node] = ind
        next_node = self.heap[next_ind][1]
        self.indices[next_node] = next_ind

        self.heapify_down(next_ind)

    # Changes the delta value of a node_id by change_in_delta amount.
    def modify_node(self, node_id, change_in_delta):
        node_ind = self.indices[node_id]
        self.heap[node_ind][0] += change_in_delta
        if change_in_delta == 0:
            return
        elif change_in_delta < 0:
            self.heapify_down(node_ind)
        else:
            self.heapify_up(node_ind)

    # Removes and returns the maximum node in the heap.
    def pop(self):
        top = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify_down(0)
        return top
    
    # Verifies that the heap is a max heap.
    def verify_heap(self, ind=0, parent=0):
        if ind >= len(self.heap):
            return True
        if self.heap[ind] > self.heap[parent]:
            return False
        right = (ind+1)*2
        left = right-1
        return self.verify_heap(left, ind) and self.verify_heap(right, ind)
    
    # Verifies that the indices match their location within the heap.
    def verify_indices(self):
        for indi, i in enumerate(self.heap):
            if not self.indices[i[1]] == indi:
                return False
        return True


# Imports the dataset csv file and returns it in the format of a list of [input, output, weight] for each connection.
def import_dataset():
    weights_df = pd.read_csv('connectome_graph.csv')
    weights_df = weights_df.sample(frac=1) #Randomizes the list
    pre_column, post_column, weight_column = 'Source Node  ID', 'Target Node ID', 'Edge Weight'
    weights_columns = [list(weights_df[x]) for x in [pre_column, post_column, weight_column]]
    weights_list = []
    for input, output, weight in zip(*weights_columns):
        weights_list.append([input, output, weight])
    return weights_list


# Returns the minimized feedback order using the GreedyFAS method. weights_list is given by import_dataset().
def get_min_feedback_order(weights_list, verbosity=0):
    all_ids = set(x[0] for x in weights_list).union(set(x[1] for x in weights_list))
    compact_ids = {neur_id: ind for ind, neur_id in enumerate(all_ids)}
    decompact_ids = {compact_ids[x]: x for x in compact_ids}
    all_nodes = set(decompact_ids.keys())

    input_dict = {x: {} for x in decompact_ids.keys()}
    output_dict = {x: {} for x in decompact_ids.keys()}

    for input, output, weight in weights_list:
        input, output = [compact_ids[x] for x in [input, output]]
        input_dict[output][input] = input_dict[output].get(input, 0) + weight
        output_dict[input][output] = output_dict[input].get(output, 0) + weight

    # Removes antiparallel edges
    for neur1 in decompact_ids:
        for neur2 in list(output_dict[neur1].keys()):
            if not neur2 in input_dict[neur1]:
                continue
            input_weight = input_dict[neur1][neur2]
            output_weight = output_dict[neur1][neur2]
            if output_weight > input_weight:
                continue

            input_weight -= output_weight
            input_dict[neur1][neur2] = input_weight
            output_dict[neur2][neur1] = input_weight
            del input_dict[neur2][neur1]
            del output_dict[neur1][neur2]
            if input_weight == 0:
                del input_dict[neur1][neur2]
                del output_dict[neur2][neur1]
            

    buckets = {x: set([]) for x in ['sources', 'sinks', 'middle']}

    def get_node_type(node):
        if not output_dict.get(node):
            return 'sinks'
        if not input_dict.get(node):
            return 'sources'
        return 'middle'
    
    def remove_from_buckets(node):
        for bucket in buckets:
            buckets[bucket].discard(node)

    def update_buckets(node):
        node_type = get_node_type(node)
        if node in buckets[node_type]:
            return
        remove_from_buckets(node)
        buckets[node_type].add(node)
    
    #Keeps track of the delta value of each neuron in a max heap.
    max_heap = MaxHeap()
    for node in decompact_ids:
        temp_delta = sum(output_dict[node].values()) - sum(input_dict[node].values())
        max_heap.add_to_heap([temp_delta, node])
        update_buckets(node)

    # Verifies that the input_dict matches the output_dict and that the heap is correct.
    def verify_data():
        for post in input_dict:
            for pre in input_dict[post]:
                assert output_dict[pre][post] == input_dict[post][pre]
        for pre in output_dict:
            for post in output_dict[pre]:
                assert output_dict[pre][post] == input_dict[post][pre]
        assert max_heap.verify_heap()
        assert max_heap.verify_indices()

    if verbosity>0:
        verify_data()

    left_list = []
    right_list = []

    # Removes a node from the buckets, input_dict, output_dict, and all_nodes.
    def remove_node(node):
        remove_from_buckets(node)
        input_neurons = input_dict[node]
        for neur in input_neurons:
            change_in_delta = -output_dict[neur][node]
            max_heap.modify_node(neur, change_in_delta)
            del output_dict[neur][node]
            update_buckets(neur)
        del input_dict[node]

        output_neurons = output_dict[node]
        for neur in output_neurons:
            change_in_delta = input_dict[neur][node]
            max_heap.modify_node(neur, change_in_delta)
            del input_dict[neur][node]
            update_buckets(neur)
        del output_dict[node]

        all_nodes.remove(node)

    # Implements GreedyFAS.
    while all_nodes:
        if buckets['sources']:
            temp_source = buckets['sources'].pop()
            remove_node(temp_source)
            left_list.append(temp_source)
            continue
        if buckets['sinks']:
            temp_sink = buckets['sinks'].pop()
            remove_node(temp_sink)
            right_list.append(temp_sink)
            continue

        _, temp_node = max_heap.pop()

        # Passes on removing the node if it has already been seen as a source or sink.
        if not temp_node in all_nodes:
            continue
        remove_node(temp_node)
        left_list.append(temp_node)

    right_list.reverse()
    neur_order = [decompact_ids[x] for x in left_list + right_list]
    return neur_order


def get_feedforward_count(weights_list, neur_order):
    all_ids = set(x[0] for x in weights_list).union(set(x[1] for x in weights_list))
    output_dict = {x: {} for x in all_ids}
    for input, output, weight in weights_list:
        output_dict[input][output] = output_dict[input].get(output, 0) + weight

    seen_neurs = set([])
    total_forward = 0
    for neur in neur_order:
        seen_neurs.add(neur)
        for output in output_dict[neur]:
            if output in seen_neurs:
                continue
            total_forward += output_dict[neur][output]
    return total_forward


def export_order(neur_order):
    order_dict = {'Node ID': [x for x in neur_order], 
                  'Order': [x for x in range(len(neur_order))]}
    order_df = pd.DataFrame(order_dict)
    file_name = os.path.join(os.getcwd(), 'feedforward_order.csv')
    order_df.to_csv(file_name)


if __name__ == '__main__':
    weights_list = import_dataset()
    #beginning = time.time()
    neur_order = get_min_feedback_order(weights_list)
    #print(time.time()-beginning)
    print('Feedforward count:', get_feedforward_count(weights_list, neur_order))
    #export_order(neur_order)

