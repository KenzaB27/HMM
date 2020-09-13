#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np
import utils as ut
import sys
from collections import namedtuple
from difflib import SequenceMatcher

THRESHOLD_STEP = 50

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.A = []
        for i in range(N_SPECIES):
            r = np.random.dirichlet(np.ones(N_SPECIES), size=1)[0]
            for j in range(N_SPECIES):
                r[j] *= 0.3
            r[i] += 0.7
            self.A.append(r)

        self.B = [np.random.dirichlet(np.ones(N_EMISSIONS), size=1)[0]
                  for _ in range(N_SPECIES)]
        self.pies = [np.random.dirichlet(np.ones(N_SPECIES), size=1)[
            0] for _ in range(N_FISH)]
        
        self.observations = [[] for _ in range(N_FISH)]
        self.groups = dict()
        self.training_set = []

    def match_maximum(self, sequences, min_length, additional_round):
        Fishes = namedtuple('Fishes', ['match_index', 'length', 'pattern'])
        best_matches = {}
        for i in range(len(sequences) - 1):
            for j in range(i + 1, len(sequences)):
                pattern = ut.match_pattern(sequences[i], sequences[j], min_length)
                if not pattern:
                    continue
                
                if i not in best_matches:
                    best_matches[i] = Fishes(-1, 0, tuple())
                if len(pattern) > best_matches[i].length:
                    best_matches[i] = Fishes(j, len(pattern), tuple(pattern))

        groups = {}
        for key_i in best_matches:
            key_j = best_matches[key_i].match_index
            pattern_i = best_matches[key_i].pattern

            if key_j not in best_matches or best_matches[key_j].length < best_matches[key_i].length:
                if pattern_i not in groups:
                    groups[pattern_i] = set()
                groups[pattern_i] = groups[pattern_i].union([key_i, key_j])
            elif additional_round:
                pattern_j = best_matches[key_j].pattern
                any_match = ut.match_pattern(pattern_i, pattern_j, min_length / 2)
                if any_match:
                    if pattern_j not in groups:
                        groups[pattern_j] = set()
                    groups[pattern_j] = groups[pattern_j].union([key_i, key_j, best_matches[key_j].match_index])
                else:
                    if pattern_i not in groups:
                        groups[pattern_i] = set()
                    groups[pattern_i].add(key_i)
            else:
                if pattern_i not in groups:
                    groups[pattern_i] = set()
                groups[pattern_i].add(key_i)
        return groups


    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        if step < THRESHOLD_STEP:
            for i in range(N_FISH):
                self.observations[i].append(observations[i])

        if step == THRESHOLD_STEP:
            self.groups = self.match_maximum(self.observations, 10, True)

            unseen_fish = set(range(N_FISH))
            for value in self.groups.values():
                self.training_set.append(self.observations[list(value)[0]])
                for i in value:
                    unseen_fish.discard(i)

            for i in unseen_fish:
                self.training_set.append(self.observations[i])
            print(len(self.training_set))
            
            print(self.A)
            print()
            print(self.B)
            print()
            print(self.pies)
            self.A, self.B, self.pies[0] = ut.baum_welch(
                self.A, self.B, self.pies[0], self.training_set[0])
        
        # if step >= THRESHOLD_STEP + 1:
            for i in range(1, len(self.training_set)):
                self.A, self.B, self.pies[i] = ut.baum_welch(self.A, self.B, self.pies[i-1], self.training_set[i])
            print()
            print()
            print(self.A)
            print()
            print(self.B)
            print()
            print(self.pies)
        
        if step == THRESHOLD_STEP + 1:
            # for i in range(10):
            path = ut.viterbi(self.A, self.B, self.pies[0], self.observations[0])
            state = max(path, key=path.count)
            print('fish', 0, path, 'state', state, file=sys.stderr)


            # i = None
            # for key in self.groups:
            #     if self.groups[key]:
            #         i = self.groups[key].pop()
            #         break
            # state = 0

            # print('Guessing fish', i, 'to be type', state)
            return 0, state
        if step == THRESHOLD_STEP + 2:
            # for i in range(10):
            path = ut.viterbi(self.A, self.B, self.pies[13], self.observations[13])
            state = max(path, key=path.count)
            print('fish', 13, path, 'state', state, file=sys.stderr)


            # i = None
            # for key in self.groups:
            #     if self.groups[key]:
            #         i = self.groups[key].pop()
            #         break
            # state = 0

            # print('Guessing fish', i, 'to be type', state)
            return 13, state

        return None

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
        # if correct:
        #     print('!! Correctly guessed that fish', fish_id, 'is type', true_type)
        # else:
        #     print('Failed to tell that fish', fish_id, 'is type', true_type)
        print('Fish', fish_id, 'is type', true_type)
