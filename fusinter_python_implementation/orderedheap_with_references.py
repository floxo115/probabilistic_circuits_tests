from typing import List


class OrderedHeapElement:
    def __init__(self, previous, sort_values: List[float], data: any = None):
        self.previous: OrderedHeapElement = previous
        self.next: OrderedHeapElement = None
        self.sort_values = sort_values
        self.data = data
        self.pos = -1

    def __lt__(self, other):
        for i, val in enumerate(self.sort_values):
            if val < other.sort_values[i]:
                return True
            if val == other.sort_values[i]:
                continue
            else:
                break

        return False

    def __gt__(self, other):
        for i, val in enumerate(self.sort_values):
            if val > other.sort_values[i]:
                return True
            if val == other.sort_values[i]:
                continue
            else:
                break

        return False

    def __repr__(self):
        return f"<OrderedHeapElement [previous_idx: {self.previous_idx}, next_idx: {self.next_idx}, value: {self.value}]>"


class OrderedHeap():
    def __init__(self):
        self.array: List[OrderedHeapElement] = []

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        s = f"<OrderedHeap [\n"
        for el in self.array:
            s += f"\t{el}\n"
        s += f"]>"
        return s


class OrderedHeap():
    def __init__(self):
        self.array: List[OrderedHeapElement] = []
        self.last_inserted: OrderedHeapElement = None
        self.first_inserted: OrderedHeapElement = None

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        s = f"<OrderedHeap [\n"
        for el in self.array:
            s += f"\t{el}\n"
        s += f"]>"
        return s

    @staticmethod
    def get_parent(pos: int):
        assert pos >= 0

        if pos == 0:
            return None
        if pos % 2 == 0:
            return (pos - 1) // 2
        else:
            return pos // 2

    @staticmethod
    def get_left_child(pos):
        assert pos >= 0
        return 2 * pos + 1

    def is_leaf(self, pos):
        return OrderedHeap.get_left_child(pos) >= len(self)

    def swap_elements(self, child_pos):
        # if pos_child is the root position raise a ValueError
        if child_pos == 0:
            raise ValueError("The root element can not be swapped with its parent!")

        # if child pos is out of bounds, raise ValueError
        if not(0 < child_pos < len(self)):
            raise ValueError("The child position is out of range")

        parent_pos = OrderedHeap.get_parent(child_pos)
        self.array[child_pos], self.array[parent_pos] = self.array[parent_pos], self.array[child_pos]

        self.array[parent_pos].pos = parent_pos
        self.array[child_pos].pos = child_pos
        return child_pos

    def bubble_up(self, pos):
        # do nothing for the root element
        if pos == 0:
            return pos

        # swap position with parent elment as long as it is smaller than elment at pos
        parent_pos = OrderedHeap.get_parent(pos)
        if self.array[parent_pos] < self.array[pos]:
            self.swap_elements(pos)
            # do the same thing recursively with new position
            pos = self.bubble_up(parent_pos)

        return pos

    def bubble_down(self, pos):
        # do nothing for leaf elements
        if self.is_leaf(pos):
            return pos

        left_child_pos = OrderedHeap.get_left_child(pos)

        # if there are two children we want to find the position of the larger one
        max_child_pos = left_child_pos
        right_child_pos = left_child_pos + 1
        if right_child_pos < len(self) and self.array[left_child_pos] < self.array[right_child_pos]:
            max_child_pos = right_child_pos

        # swap position if the larger child is larger than element at position
        if self.array[max_child_pos] > self.array[pos]:
            self.swap_elements(max_child_pos)
            # do the same thing recursively with new position
            pos = self.bubble_down(max_child_pos)


    def insert_element(self, sort_values, data=None):
        previous = self.last_inserted

        new_element = OrderedHeapElement(previous, sort_values, data=data)
        new_element.pos = len(self)
        if self.first_inserted is None:
            self.first_inserted = new_element

        if previous is not None:
            previous.next = new_element
        self.array.append(new_element)

        self.last_inserted = new_element

    def build_heap(self):
        # TODO the first index can be tightened
        len_heap = len(self)
        for pos in range(len_heap, - 1, -1):
            #print("heap build pos", pos, flush=True)
            self.bubble_down(pos)

    def delete_max(self):
        if len(self) == 0:
            raise ValueError("The heap is already empty")

        node_to_delete = self.array[0]

        # fix the node references
        node_prev = node_to_delete.previous
        node_next = node_to_delete.next

        if node_prev:
            node_prev.next = node_to_delete.next
        else:
            self.first_inserted = node_next
        if node_next:
            node_next.previous = node_to_delete.previous
        else:
            self.last_inserted = node_prev

        # put last element in array in first position and delete last position
        self.array[0] = self.array[-1]
        self.array[0].pos = 0
        self.array.pop()

        self.bubble_down(0)

    def update_node(self, new_sort_values, pos):
        self.array[pos].sort_values = new_sort_values
        pos = self.bubble_up(pos)
        pos = self.bubble_down(pos)

