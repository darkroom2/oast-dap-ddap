from typing import List


class Path:
    def __init__(self, _id: int, link_ids: List[int]):
        self.id = _id
        self.link_ids = link_ids

    def __repr__(self):
        return f'{self.id=}, {self.link_ids=}'


class Demand:
    def __init__(
            self,
            _id: int,
            start_node: int,
            end_node: int,
            volume: int,
            paths: List[Path]
    ):
        self.id = _id
        self.start_node = start_node
        self.end_node = end_node
        self.volume = volume
        self.paths = paths

    def __repr__(self):
        return f'{self.id=}, {self.start_node=}, {self.end_node=}, {self.volume=}, {self.paths=}'


class Link:
    def __init__(
            self,
            _id: int,
            start_node: int,
            end_node: int,
            number_of_modules: int,
            module_cost: int,
            link_module: int
    ):
        self.id = _id
        self.start_node = start_node
        self.end_node = end_node
        self.number_of_modules = number_of_modules
        self.module_cost = module_cost
        self.link_module = link_module

    def __repr__(self):
        return f'{self.id=}, {self.start_node=}, {self.end_node=}, {self.number_of_modules=}, {self.module_cost=}, {self.link_module=} '


class Network:
    def __init__(self, links: List[Link], demands: List[Demand]):
        self.links = links
        self.demands = demands

    def __repr__(self):
        return f'{self.links=}, {self.demands=}'
