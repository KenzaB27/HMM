#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np
import utils as ut
import sys
from difflib import SequenceMatcher

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.A = [np.random.dirichlet(np.ones(N_SPECIES), size=1)[
            0] for _ in range(N_SPECIES)]
        self.B = [np.random.dirichlet(np.ones(N_EMISSIONS), size=1)[0]
                  for _ in range(N_SPECIES)]
        self.pies = [np.random.dirichlet(np.ones(N_SPECIES), size=1)[
            0] for _ in range(N_FISH)]
        
        self.observations = [[] for _ in range(N_FISH)]
        pass

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        if step < 20:
            for i in range(N_FISH):
                self.observations[i].append(observations[i])

        if step == 20:
            observations_clusters = []
            for i in range(20):
                threshold = 12
                pairs = [i]
                for j in range(i,20):
                    s = SequenceMatcher(None, self.observations[i], self.observations[j])
                    match = s.find_longest_match(0, 19, 0, 19)
                    if match.size >= threshold:
                        pairs.append(j)
                observations_clusters.append(pairs)
            self.A, self.B, self.pies[0] = ut.baum_welch(
                self.A, self.B, self.pies[0], self.observations[0])
            for i in range(1,10):
                self.A, self.B, self.pies[i] = ut.baum_welch(self.A, self.B, self.pies[i-1], self.observations[i])
        if step == 21:
            for i in range(10):
                path = ut.viterbi(self.A, self.B, self.pies[0], self.observations[0])
                print('fish', i, path, 'state', max(
                    path, key=path.count), file=sys.stderr)

        return None

        # This code would make a random guess on each step:

        # return None

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        pass
