from gym import spaces
import numpy as np
from heapq import nlargest
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

class FreeRENAS_RecombinationEnv(BaseNASEnv):
    """
    gym.Env for FreeRENAS algorithm, RENAS using Training-Free metrics.
    In this environment we present an integration of DRL with Genetic Computations that handles
    recombination operations rather than mutation only.

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
        # the number of individuals the agent is called to perform recombination with
        self.n_parents = genetics_dict.get("n_parents", 2)
        """
        Each individual has a genotype of lenght `m` (clearly enough, `m = m(searchspace)`). Each
        individual gene can take any of the `n` operations in the search-space genome (the set of 
        all possible operations), thus each individual can be represented via integer encoding.
        In this case, observations would be `self.n_parents x (m,)` tensors whose values would range 
        in [0, n-1] and represent one of the different `n` genes for each parent, repeated
        `self.n_parents` times vertically.
        """
        self.observation_space = spaces.MultiDiscrete(
            [len(self.searchspace.all_ops) for _ in range(self.searchspace.architecture_len)] \
                * self.n_parents
        )
        """
        The controller interaction with the various parents happens choosing which genes to select from
        each parent. 
        Therefore, it chooses, for each of the `n` available genes in the child individual the parent
        to take the same gene from.
        """
        # single action = [(parent_1), (parent_2), ... (parent_2)]
        self.action_space = spaces.MultiDiscrete(
            [self.n_parents for _ in range(self.searchspace.architecture_len)]
        )
        
        # initializes population, timestep counter and current maximal fitness
        self.reset()

    @property
    def name(self)->Text:
        return "freerenas-recombination"

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

    def perform_recombination(self, parents:Iterable[NDArray], recombination_idx:NDArray):
        
        merged = np.array(parents)
        return merged[recombination_idx, np.arange(len(recombination_idx))]

    def _get_obs(self)->NDArray: 
        """Return current observation as defined at observation space level."""
        # create the tournament
        tournament = self.get_tournament(population=self.population.individuals)
        # select the fittest individuals in the tournament considered
        parents = nlargest(n=self.n_parents, iterable=tournament, key=lambda i: i.fitness)
        # saving parents
        self.parents = parents
        # numerically encode the parents in the tournament (the one that will undergo mutation!)
        encoded_parents = [self.searchspace.encode_architecture(
            architecture_string = self.searchspace.list_to_architecture(ind.genotype), 
            onehot = False
        ) for ind in parents]
        # [(parent_1), (parent_2)] -> [parent_1_gene1, ..., parent_1_genen, parent_2_gene1, ..., parent_2_genen]
        return np.concatenate(encoded_parents)
    
    def _get_info(self)->dict: 
        """Return the info dictionary."""
        info_dict = {
            "parents_genotypes": list(map(lambda ind: ind.genotype, self.parents)),
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
    
    def get_reward(self, recombinant:BaseIndividual)->float:
        """
        Compute the reward associated to the mutation operation.
        Here, the reward is defined as the gain in fitness between the parent and mutant invidual.s
        """
        fittest_parent = max(self.parents, key=lambda ind: ind.fitness)
        recombinant_fitness, fittest_parent_fitness = recombinant.fitness, fittest_parent.fitness
        # mutation increasing the fitness value should be rewarded more
        return recombinant_fitness - fittest_parent_fitness
    
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

        # genotype -> architecture string -> architecture encoding
        parents_encoded = [self.searchspace.encode_architecture(
            architecture_string=self.searchspace.list_to_architecture(parent.genotype),
            onehot=False)
            for parent in self.parents
        ]
        
        # applies recombination of parents
        recombinant_encoded = self.perform_recombination(parents=parents_encoded, 
                                                         recombination_idx=action)
        
        # one-hot like actions actually represent individuals when coming from OneHot space
        reinforced_mutant = self.searchspace.decode_architecture(
            architecture_encoded=recombinant_encoded,
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
        reward = self.get_reward(recombinant=reinforced_individual)

        # check whether or not the episode is terminated
        terminated = self.is_done()
        
        # obtain obs and info
        observation = self._get_obs()
        info = self._get_info()

        return (observation,
                reward,
                terminated,
                info)
