from itertools import product, chain
from json import loads, dumps
from math import ceil
from multiprocessing import Pool
from pathlib import Path as PathLib
from random import choice, shuffle, seed, random, sample
from time import perf_counter
from typing import Tuple

from lxml import etree

from src.network import Path, Demand, Link, Network
from src.utils import setup_logger

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
        self.genes = 0

    def add_gene(self, gene):
        self.allocation_pattern.update(gene)
        self.genes += 1

    def build_chromosome(self, all_solutions):
        for solution in all_solutions:
            self.add_gene(choice(solution))

    def get_gene(self, gene_id):
        return {key: value for key, value in self.allocation_pattern.items() if
                key[0] == gene_id}

    def mutate_gene(self, gene_number):
        gene = self.get_gene(gene_number)
        if len(gene) > 1:
            flows = sample(list(gene), 2)
            if self.allocation_pattern[flows[0]] > 0:
                self.allocation_pattern[flows[0]] -= 1
                self.allocation_pattern[flows[1]] += 1


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
            self.all_solutions = pool.map(self.get_valid_solutions_for_demand,
                                          self.network.demands)

    def get_links_for_alloc(self, allocation_pattern):
        raise NotImplementedError()

    def calculate_z(self, link_values):
        raise NotImplementedError()


class DAP(Problem):
    def calculate_z(self, link_values):
        z = float('-inf')
        for link_id, link_load in enumerate(link_values):
            _z = link_load - self.network.links[link_id].number_of_modules * \
                 self.network.links[link_id].link_module
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
    def __init__(self, problem_instance: Problem, _seed: int,
                 number_of_chromosomes: int, stop_criterion: str, max_time: int,
                 max_generations: int, max_mutations: int,
                 max_no_progress_gen: int, crossover_prob: float,
                 mutation_prob: float):
        self.problem_instance = problem_instance
        self.seed = _seed
        self.number_of_chromosomes = number_of_chromosomes
        self.stop_criterion = stop_criterion
        self.current_generation = 0
        self.mutations = 0
        self.start_time = 0
        self.no_progress_gen = 0
        self.max_time = max_time
        self.max_generations = max_generations
        self.max_mutations = max_mutations
        self.max_no_progress_gen = max_no_progress_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elapsed_time = 0

    def compute(self):
        seed(self.seed)

        self.start_time = perf_counter()

        population = self.init_pop()

        population = map(self.calc_fitness, population)
        population = sorted(population, key=lambda x: x.z)
        solution = population[0]

        while not self.end_condition():
            logger.info(
                f'Generation: {self.current_generation}, goal: {solution.z}\n\t'
                f'solution.link_values: {solution.link_values}\n\t'
                f'solution.link_values: {solution.allocation_pattern}')
            # choose better half to be parents
            parents = population[:len(population) // 2]
            # make random pairs
            shuffle(parents)
            parent_pairs = zip(parents[0::2], parents[1::2])
            # generate offspring
            offspring_pairs = map(self.generate_offspring, parent_pairs)

            offsprings = map(self.calc_fitness,
                             chain.from_iterable(offspring_pairs))

            population.extend(offsprings)
            population = sorted(population, key=lambda x: x.z)[
                         :self.number_of_chromosomes]
            best_chromosome = population[0]

            if best_chromosome.z < solution.z:
                solution = best_chromosome
                self.no_progress_gen = 0
            else:
                self.no_progress_gen += 1

            self.current_generation += 1
        self.elapsed_time = perf_counter() - self.start_time
        logger.info(f'Took {self.elapsed_time} seconds...')
        return solution

    def calc_fitness(self, chromosome):
        chromosome.link_values = self.problem_instance.get_links_for_alloc(
            chromosome.allocation_pattern)
        chromosome.z = self.problem_instance.calculate_z(chromosome.link_values)
        return chromosome

    def prepare_chromosome(self, _):
        chromosome = Chromosome()
        chromosome.build_chromosome(self.problem_instance.all_solutions)
        return chromosome

    def generate_chromosomes(self):
        return list(
            map(self.prepare_chromosome, range(self.number_of_chromosomes)))

    def init_pop(self):
        chromosomes = self.generate_chromosomes()
        shuffle(chromosomes)
        return chromosomes

    def end_condition(self):
        if self.stop_criterion == 'time':
            if perf_counter() - self.start_time > self.max_time:
                logger.info('Max time exceeded, stopping...')
                return True

        elif self.stop_criterion == 'max_gen':
            if self.current_generation > self.max_generations:
                logger.info('Max generations exceeded, stopping...')
                return True

        elif self.stop_criterion == 'no_progress':
            if self.no_progress_gen > self.max_no_progress_gen:
                logger.info(
                    'Max generations without progress exceeded, stopping...')
                return True

        elif self.stop_criterion == 'max_mut':
            if self.max_mutations == 0:
                logger.info('Max mutations performed exceeded, stopping...')
                return True

        else:
            return False

    def generate_offspring(self, parents: Tuple[Chromosome, Chromosome]):
        parent_1, parent_2 = parents
        child_1 = Chromosome()
        child_2 = Chromosome()

        # crossover
        for gene_id in range(1, len(self.problem_instance.network.demands) + 1):
            if random() < self.crossover_prob:
                child_1.add_gene(parent_1.get_gene(gene_id))
                child_2.add_gene(parent_2.get_gene(gene_id))
            else:
                child_1.add_gene(parent_2.get_gene(gene_id))
                child_2.add_gene(parent_1.get_gene(gene_id))

        # mutation
        for child in [child_1, child_2]:
            if random() < self.mutation_prob:
                for i in range(child.genes):
                    if random() < self.mutation_prob:
                        child.mutate_gene(i + 1)
                        self.mutations += 1

        return child_1, child_2


class BruteForceAlgorithm:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance
        self.start_time = 0
        self.elapsed_time = 0

    def calc_fitness(self, chromosome):
        chromosome.link_values = self.problem_instance.get_links_for_alloc(
            chromosome.allocation_pattern)
        chromosome.z = self.problem_instance.calculate_z(chromosome.link_values)
        return chromosome

    def compute(self):
        start_time = perf_counter()
        solutions = self.problem_instance.all_solutions

        valid_solutions = map(self.make_solution, product(*solutions))
        valid_solutions = map(self.calc_fitness, valid_solutions)
        valid_solutions = sorted(valid_solutions, key=lambda x: x.z)
        self.elapsed_time = perf_counter() - start_time
        logger.info(f'Brute Force took: {self.elapsed_time} s.')
        return valid_solutions[0]

    @staticmethod
    def make_solution(aloc_pattern: dict):
        chromosome = Chromosome()
        for value in aloc_pattern:
            chromosome.add_gene(value)
        return chromosome


def main():
    # Wyniki z AMPL:
    # net4:
    #   * DAP = -138
    #   * DDAP = 13
    # net12_1:
    #   * DAP = -29
    #   * DDAP = 26
    # net12_2:
    #   * DAP = 0
    #   * DDAP = 32

    # WCZYTANIE PARAMETROW
    config = loads(PathLib('config/config.json').read_bytes())

    # PARSOWANIE SIECI Z PLIKU DO OBIEKTU
    network_path = PathLib(config.get('network_path'))
    network = parse_xml_network(str(network_path.resolve()))
    problem_type = config.get('problem_type')
    algorithm_type = config.get('algorithm_type')
    # WYBOR PROBLEMU
    if problem_type == 'DAP':
        problem_instance = DAP(network)
    else:
        problem_instance = DDAP(network)

    # GENEROWANIE WSZYSTKICH MOŻLIWYCH ROZWIĄZAŃ
    problem_instance.generate_all_valid_solutions()

    # TODO: BRUTE FORCE ALGORITHM DO ZROBIENIA
    if algorithm_type == 'EA':
        algorithm = EvolutionaryAlgorithm(
            problem_instance=problem_instance,
            _seed=config.get('seed'),
            number_of_chromosomes=config.get('pop'),
            stop_criterion=config.get('stop_criterion'),
            max_time=config.get('max_time'),
            max_mutations=config.get('max_mutations'),
            max_generations=config.get('max_generations'),
            max_no_progress_gen=config.get('max_no_progress_gen'),
            crossover_prob=config.get('crossover_prob'),
            mutation_prob=config.get('mutation_prob')
        )
    else:
        algorithm = BruteForceAlgorithm(
            problem_instance=problem_instance
        )

    # ROZWIAZANIE
    solution = algorithm.compute()
    logger.info(f'Solution:\n\t {solution.allocation_pattern}\n\twith link '
                f'values: {solution.link_values}\n\tand cost: {solution.z} '
                f'{problem_type} / {config.get("network_path")} using: '
                f'{algorithm_type}')

    curr_gen = algorithm.current_generation if algorithm_type == 'EA' else 0

    obciazenie = [elem * network.links[0].link_module for elem in
                  solution.link_values]

    res = {
        'fn_kosztu': solution.z,
        'iteracje': curr_gen,
        'czas_opt': algorithm.elapsed_time,
        'liczn_pop': config.get('pop'),
        'prawd_krzyz': config.get('crossover_prob'),
        'prawd_mut': config.get('mutation_prob'),
        'obciazenie': obciazenie,
        'wymiary': solution.link_values,
        'rozklad_zapotrz': f'{solution.allocation_pattern}'
    }

    PathLib(f'results/{problem_type}_{algorithm_type}_{network_path.stem}_'
            f'result.json').write_text(dumps(res, indent=4))


if __name__ == '__main__':
    main()
