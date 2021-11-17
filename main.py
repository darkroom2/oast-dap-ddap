from itertools import product, chain
from math import ceil
from multiprocessing import Pool
from random import choice, shuffle, seed, getrandbits
from time import perf_counter
from typing import Tuple

from lxml import etree

from network import Path, Demand, Link, Network
from utils import setup_logger

logger = setup_logger()


def parse_xml_network(network_path):
    demands = []
    links = []
    tree = etree.parse(network_path)
    network_xml = tree.getroot()
    links_xml, demands_xml = network_xml.getchildren()
    for demand in demands_xml:
        demand_id = int(demand.get('id'))
        start_node = int(demand.find('startNode').text)
        end_node = int(demand.find('endNode').text)
        volume = int(demand.find('volume').text)
        paths = []
        for path in demand.find('paths'):
            path_id = int(path.get('id'))
            link_ids = [int(link_id.text) for link_id in path.iter('linkId')]
            paths.append(Path(path_id, link_ids))
        demands.append(Demand(demand_id, start_node, end_node, volume, paths))
    for link in links_xml:
        link_id = int(link.get('id'))
        start_node = int(link.find('startNode').text)
        end_node = int(link.find('endNode').text)
        number_of_modules = int(link.find('numberOfModules').text)
        module_cost = int(link.find('moduleCost').text)
        link_module = int(link.find('linkModule').text)
        links.append(
            Link(
                link_id,
                start_node,
                end_node,
                number_of_modules,
                module_cost,
                link_module
            )
        )

    return Network(links, demands)


class Chromosome:
    def __init__(self):
        self.allocation_pattern = {}
        self.link_values = []
        self.z = float('inf')

    def __repr__(self):
        return f'{self.allocation_pattern=}'

    def add_gene(self, gene):
        self.allocation_pattern.update(gene)

    def build_chromosome(self, all_solutions):
        for solution in all_solutions:
            self.add_gene(choice(solution))

    def get_gene(self, gene_id):
        return {key: value for key, value in self.allocation_pattern.items() if key[0] == gene_id}


class Problem:
    def __init__(self, network: Network):
        self.network = network
        self.all_solutions = []

    @staticmethod
    def get_valid_solutions_for_demand(demand):
        demand_solutions = []

        all_combinations = product(
            range(demand.volume + 1),
            repeat=len(demand.paths)
        )

        valid_combinations = filter(
            lambda x: sum(x) == demand.volume,
            all_combinations
        )

        for combination in valid_combinations:
            allocation_vector = {}
            for demand_path in demand.paths:
                flow_xdp = (demand.id, demand_path.id)
                allocation_vector[flow_xdp] = combination[demand_path.id - 1]
            demand_solutions.append(allocation_vector)
        return demand_solutions

    def generate_all_valid_solutions(self):
        with Pool() as pool:
            self.all_solutions = pool.map(self.get_valid_solutions_for_demand, self.network.demands)

    def get_links_for_alloc(self, allocation_pattern):
        raise NotImplementedError()

    def calculate_z(self, link_values):
        raise NotImplementedError()


class DAP(Problem):
    def calculate_z(self, link_values):
        z = float('-inf')
        for link_id, link_load in enumerate(link_values):
            _z = link_load - self.network.links[link_id].number_of_modules * self.network.links[link_id].link_module
            if _z > z:
                z = _z
        return z

    def get_links_for_alloc(self, allocation_pattern):
        links = self.network.links
        link_values = [0] * len(links)
        for link in links:
            volume_sum = 0
            for demand in self.network.demands:
                for path in demand.paths:
                    if link.id in path.link_ids:
                        volume = allocation_pattern.get((demand.id, path.id))
                        volume_sum += volume
            link_values[link.id - 1] = volume_sum
        return link_values


class DDAP(Problem):
    def calculate_z(self, link_values):
        z = 0
        for link_id, link_size in enumerate(link_values):
            z += self.network.links[link_id].module_cost * link_size
        return z

    def get_links_for_alloc(self, allocation_pattern):
        links = self.network.links
        link_values = [0] * len(links)
        for link in links:
            volume_sum = 0
            for demand in self.network.demands:
                for path in demand.paths:
                    if link.id in path.link_ids:
                        volume = allocation_pattern.get((demand.id, path.id))
                        volume_sum += volume
            link_values[link.id - 1] = ceil(volume_sum / link.link_module)
        return link_values


class EvolutionaryAlgorithm:
    def __init__(self, problem_instance: Problem, seed: int, number_of_chromosomes: int, max_time: int,
                 max_generations: int, max_mutations: int, max_no_progress_gen: int, crossover_prob: float,
                 mutation_prob: float
                 ):
        self.problem_instance = problem_instance
        self.seed = seed
        self.number_of_chromosomes = number_of_chromosomes
        self.current_generation = 0
        self.start_time = 0
        self.no_progress_gen = 0
        self.max_time = max_time
        self.max_generations = max_generations
        self.max_mutations = max_mutations
        self.max_no_progress_gen = max_no_progress_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def compute(self):
        seed(self.seed)

        self.start_time = perf_counter()

        population = self.init_pop()

        population = map(self.calc_fitness, population)
        population = sorted(population, key=lambda x: x.z)
        solution = population[0]

        while not self.end_condition():
            logger.info(f'Generation: {self.current_generation}, goal: {solution.z}')
            # choose better half to be parents
            parents = population[:len(population) // 2]
            # make random pairs
            shuffle(parents)
            parent_pairs = zip(parents[0::2], parents[1::2])
            # generate offspring
            offspring_pairs = map(self.generate_offspring, parent_pairs)
            offsprings = map(self.calc_fitness, chain.from_iterable(offspring_pairs))

            population.extend(offsprings)
            population = sorted(population, key=lambda x: x.z)[:self.number_of_chromosomes]
            best_chromosome = population[0]

            if best_chromosome.z < solution.z:
                solution = best_chromosome
                self.no_progress_gen = 0
            else:
                self.no_progress_gen += 1

            self.current_generation += 1

            # TODO: mutacje

        logger.info(f'Took {perf_counter() - self.start_time} seconds...')
        return solution

    def calc_fitness(self, chromosome):
        chromosome.link_values = self.problem_instance.get_links_for_alloc(chromosome.allocation_pattern)
        chromosome.z = self.problem_instance.calculate_z(chromosome.link_values)
        return chromosome

    def prepare_chromosome(self, _):
        chromosome = Chromosome()
        chromosome.build_chromosome(self.problem_instance.all_solutions)
        return chromosome

    def generate_chromosomes(self):
        return list(map(self.prepare_chromosome, range(self.number_of_chromosomes)))

    def init_pop(self):
        chromosomes = self.generate_chromosomes()
        shuffle(chromosomes)
        return chromosomes

    def end_condition(self):
        if perf_counter() - self.start_time > self.max_time:
            logger.info('Max time exceeded, stopping...')
            return True

        elif self.current_generation > self.max_generations:
            logger.info('Max generations exceeded, stopping...')
            return True

        elif self.no_progress_gen > self.max_no_progress_gen:
            logger.info('Max generations without progress exceeded, stopping...')
            return True

        elif self.max_mutations == 0:
            logger.info('Max mutations performed exceeded, stopping...')
            return True

        else:
            return False

    def generate_offspring(self, parents: Tuple[Chromosome, Chromosome]):
        parent_1, parent_2 = parents
        child_1 = Chromosome()
        child_2 = Chromosome()

        for gene_id in range(1, len(self.problem_instance.network.demands) + 1):
            if getrandbits(1):
                child_1.add_gene(parent_1.get_gene(gene_id))
                child_2.add_gene(parent_2.get_gene(gene_id))
            else:
                child_1.add_gene(parent_2.get_gene(gene_id))
                child_2.add_gene(parent_1.get_gene(gene_id))
        return child_1, child_2


def main():
    input_path = 'networks/net12_2.xml'
    # input_path = 'networks/net4.xml'
    algorithm_type = ['EA', 'BFA'][0]
    problem_type = ['DAP', 'DDAP'][1]

    network = parse_xml_network(input_path)

    if problem_type == 'DAP':
        problem_instance = DAP(network)
    else:
        problem_instance = DDAP(network)

    problem_instance.generate_all_valid_solutions()

    if algorithm_type == 'EA':
        algorithm = EvolutionaryAlgorithm(
            problem_instance=problem_instance,
            seed=123,
            number_of_chromosomes=1000,
            max_time=300,
            max_mutations=100,
            max_generations=100,
            max_no_progress_gen=20,
            crossover_prob=0.6,
            mutation_prob=0.1
        )
    else:
        algorithm = None

    solution = algorithm.compute()
    logger.info(f'Solution: {solution.link_values}')


if __name__ == '__main__':
    main()
