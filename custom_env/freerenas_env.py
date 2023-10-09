from gym import spaces
import numpy as np
from itertools import chain
from numpy.typing import NDArray
from .base_env import (
    BaseNASEnv
)
from commons import (
    BaseInterface,
    Population,
    BaseIndividual,
    FastIndividual,
    Individual,
)
from typing import Iterable, Tuple, Text

class FreeRENASEnv(BaseNASEnv): 
    """
    gym.Env for FreeRENAS algorithm, RENAS using Training-Free metrics.
    In this environment we present an integration of DRL with Genetic Computations.

    The RENAS paper can be found at: https://arxiv.org/abs/1808.00193
    The FreeREA paper can be found at: https://arxiv.org/pdf/2207.05135.pdf
    """
    def __init__(self, 
                 searchspace_api:BaseInterface,
                 genetics_dict:dict={}, 
                 scores:Iterable[Text]=["naswot_score", "logsynflow_score", "skip_score"]):
        
        self.searchspace = searchspace_api
        self.fast_inds = True  # only uses fast individuals
        self.score_names = scores

        # the number of individuals in the population is set through genetics dict
        self.population_size = genetics_dict.get("population_size", 20)
        # genetics_dict also sets the number of individuals to be used in the tournament
        self.tournament_size = genetics_dict.get("tournament_size", 5)
        # the episode duration is always the number of generations the population can evolve
        self.max_generations = genetics_dict.get("max_generations", 50)
        # the number of individual mutations the agent is called to perform
        self.n_mutations = genetics_dict.get("n_mutations", 1)
        """
        The following are cross-episodes specs.
        `mut_probability = P(muation occurs)`
        """
        self.mut_probability = genetics_dict.get("mutation_probability", 1.)
        """
        Each individual has a genotype of lenght `m` (clearly enough, `m = m(searchspace)`). Each
        individual gene can take any of the `n` operations in the search-space genome (the set of 
        all possible operations), thus each individual can be represented via integer encoding.
        In this case, observations would be `(m,)` tensors whose values would range in [0, n-1] and
        represent one of the different `n` genes.
        """
        self.observation_space = spaces.MultiDiscrete(
            [len(self.searchspace.all_ops) for _ in range(self.searchspace.architecture_len)]
            )
        """
        The controller interaction with the various individuals happens choosing what locus to change
        and how to do it. Therefore, it chooses one of the different `m` loci and then selects one
        of the `n` available genes for the mutation considered. Selecting the locus `m+1` corresponds
        to not applying any change to the individual considered. Each single interaction is an individual
        mutation, whose number is expressed through `n_mutations`.
        """
        # single action = [(locus_to_mutate, gene_to_use_for_mutation)] * number_of_mutations
        action_space = list(chain(*[
            (self.searchspace.architecture_len + 1, len(self.searchspace.all_ops)) 
            for _ in range(self.n_mutations)])
        )
        self.action_space = spaces.MultiDiscrete(action_space)
        
        # initializes population, timestep counter and current maximal fitness
        self.reset()

    @property
    def name(self)->Text:
        return "freerenas"

    @property
    def fittest_individual(self)->Individual: 
        """Returns fittest individual"""
        if not hasattr(self, "_fittest_individual"): 
            self._fittest_individual = max(self.population.individuals, key=lambda ind: ind.fitness)
        
        return self._fittest_individual

    def update_fittest(self, new_fittest:BaseIndividual):
        if new_fittest.fitness >= self.fittest_individual.fitness: 
            # update new fitness
            self._fittest_individual = new_fittest
        else: 
            print(f"New ind: {new_fittest.fitness}, Current fittest: {self.fittest_individual.fitness}")
            raise ValueError("New fittest's fitness is not larger than old fitness!")
    
    @property
    def pop_fitness_mean(self)->float: 
        """Returns mean value of fitness for population"""
        return np.mean([ind.fitness for ind in self.population])
    
    @property
    def pop_fitness_std(self)->float:
        """Returns std value of fitness for population"""
        return np.std([ind.fitness for ind in self.population])

    def get_tournament(self, population:Iterable[BaseIndividual])->Tuple[BaseIndividual]:
        """
        Returns a subsample of the population from which to obtain the parents.
        In RENAS, the tournament is sampled randomly from the population.
        """
        return np.random.choice(a=population, size=self.tournament_size, replace=False).tolist()

    def normalize_score(self, score_value:float, score_name:Text, type:Text="std")->float:
        """
        Normalize the given score value using a specified normalization type.

        Args:
            score_value (float): The score value to be normalized.
            score_name (Text): The name of the score used for normalization.
            type (Text, optional): The type of normalization to be applied. Defaults to "std".

        Returns:
            float: The normalized score value.

        Raises:
            ValueError: If the specified normalization type is not available.

        Note:
            The available normalization types are:
            - "std": Standard score normalization using mean and standard deviation.
        """
        if type == "std":
            score_mean = self.searchspace.get_score_mean(score_name)
            score_std = self.searchspace.get_score_std(score_name)
            
            return (score_value - score_mean) / score_std
        else:
            raise ValueError(f"Normalization type {type} not available!")
    
    def fitness_function(self, individual:BaseIndividual)->Individual: 
        """
        Directly overwrites the fitness attribute for a given individual.
        In RENAS, the fitness score coincides with the test accuracy.

        Args: 
            individual (Individual): Individual to score.

        Returns:
            Individual: Individual, with fitness field.
        """
        if individual.fitness is None:  # None at initialization 
            scores = np.array([
                self.normalize_score(
                    score_value=self.searchspace.list_to_score(input_list=individual.genotype, score=score), 
                    score_name=score
                )
                for score in self.score_names
            ])
                        
            # individual fitness is a the sum of the various individual scores
            individual._fitness = scores.sum()
        
        return individual

    def perform_mutation(self, mutant:NDArray, mutation:Tuple[int, int])->NDArray: 
        """
        Performs mutation.

        Args:
            mutant (NDArray): The observation array to be mutated.
            mutation (Tuple[int, int]): A tuple representing the mutation locus and gene.

        Returns:
            NDArray: The mutated observation array.

        Raises:
            IndexError: If the mutation locus is outside the boundaries of the observation.

        Note:
            If the mutation locus is outside the observation boundaries, the "leave-as-is" 
            mutation action is performed.
        """
        try: 
            mutant_locus, mutant_gene = mutation
            # overwriting mutant_locus with mutant_gene
            mutant[mutant_locus] = mutant_gene
        except IndexError: 
            """
            Index error is caused by mutation[0] being outside of observation boundaries. 
            This corresponds to perform the "leave-as-is" mutation action
            """
            pass

        return mutant
        
    def _get_obs(self)->NDArray: 
        """Return current observation as defined at observation space level."""
        # create the tournament
        tournament = self.get_tournament(population=self.population.individuals)
        # select the two individuals in the tournament considered
        worst, best = min(tournament, key=lambda i: i.fitness), max(tournament, key=lambda i: i.fitness)
        self.parent = best
        # remove worst from population
        self.population.individuals.remove(worst)
        # numerically encode the best element in the tournament (the one that will undergo mutation!)
        return self.searchspace.encode_architecture(
            architecture_string = self.searchspace.list_to_architecture(best.genotype), 
            onehot = False
        )
    
    def _get_info(self)->dict: 
        """Return the info dictionary."""
        info_dict = {
            "parent": self.parent.genotype,
            "current_max_fitness": self.fittest_individual.fitness, 
            "timestep": self.timestep, 
            "mean_fitness": self.pop_fitness_mean,
            "std_fitness": self.pop_fitness_std
        }

        return info_dict

    def is_done(self)->bool: 
        """Returns `True` at episode termination and `False` before."""
        return self.timestep + 1 >= self.max_generations
    
    def reset(self)->NDArray:
        """Resets custom env attributes."""
        try: 
            del self.population
        except AttributeError: 
            """Population was already not defined - This happens only during initialization."""
            # Fail silently...
            pass
        
        # resets population
        self.population = Population(
            space = self.searchspace, 
            init_population = True, 
            n_individuals = self.population_size, 
            fast_inds = self.fast_inds
        )
        # individuals are scored according to their test accuracy
        self.population.apply_on_individuals(self.fitness_function)

        # set current fittest
        _ = self.fittest_individual

        # resets current timesteps counter
        self.timestep = 0
        # resents current max fitness considered
        self.current_max_fitness = float("-inf")

        # accessing observation and info
        observation = self._get_obs()

        # storing the buffer of all the individuals that have ever been seen during training 
        self.history = {}

        return observation
    
    def get_reward(self, mutant:BaseIndividual)->float:
        """
        Compute the reward associated to the mutation operation.
        Here, the reward is defined as the gain in fitness between the parent and mutant invidual.s
        """
        mutant_fitness, parent_fitness = mutant.fitness, self.parent.fitness
        # mutation increasing the fitness value should be rewarded more
        return mutant_fitness - parent_fitness
    
    def step(self, action:NDArray)->Tuple[NDArray, float, bool, dict]: 
        """Steps the episode having a given action.
        
        Args:
            action (NDArray): Action to be performed.
        
        Returns:
            Tuple[NDArray, float, bool, dict]: New observation (after having performed the action), 
                                               reward value,
                                               done signal (True at episode termination), 
                                               info dictionary
        """
        # saving the current population 
        self.history.update(
            {self.searchspace.list_to_architecture(ind.genotype): ind 
             for ind in self.population}
        )

        # increment timestep counter (used to declare episode termination)
        self.timestep += 1

        # split action into a list of 2-item tuples (as per the definition of action space)
        mutation_list = [(action[i], action[i+1]) for i in range(0, len(action), 2)]
        # genotype -> architecture string -> architecture encoding
        parent_encoded = self.searchspace.encode_architecture(
            architecture_string=self.searchspace.list_to_architecture(self.parent.genotype),
            onehot=False
            )
        # copying parent encoding before performing mutations - copy=True is slow, using a = I @ b instead
        mutant_encoded = np.diag(np.ones_like(parent_encoded)) @ parent_encoded
        # perform all the mutations in `mutation_list`
        for mutation in mutation_list:
            mutant_encoded = self.perform_mutation(mutant=mutant_encoded, mutation=mutation)
        
        # one-hot like actions actually represent individuals when coming from OneHot space
        reinforced_mutant = self.searchspace.decode_architecture(
            architecture_encoded=mutant_encoded, 
            onehot=False
        )
        # obtaining genotype from reinforced mutant
        reinforced_genotype = self.searchspace.architecture_to_list(
            architecture_string=reinforced_mutant
        )
        # creating new individual with reinforced-controlled genotype
        reinforced_individual = FastIndividual(genotype=None, 
                                                index=None,
                                                genotype_to_idx=self.searchspace.architecture_to_index
                                                )
        reinforced_individual.update_genotype(new_genotype=reinforced_genotype)
                
        # score individual based on its test accuracy
        reinforced_individual = self.fitness_function(reinforced_individual)
        # add reinforced individual to population (worst individual is removed in _get_obs())
        self.population.add_to_population(new_individuals=[reinforced_individual])

        # update current maximal value of fitness
        if reinforced_individual.fitness >= self.fittest_individual.fitness:
            self.update_fittest(new_fittest=reinforced_individual)

        # compute the reward associated with producing reinforced_individual
        reward = self.get_reward(mutant=reinforced_individual)

        # check whether or not the episode is terminated
        terminated = self.is_done()
        
        # obtain obs and info
        observation = self._get_obs()
        info = self._get_info()

        return (observation,
                reward,
                terminated,
                info)
