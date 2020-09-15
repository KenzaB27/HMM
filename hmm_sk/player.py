#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np
import utils as ut
import sys
from collections import namedtuple
from difflib import SequenceMatcher

THRESHOLD_STEP = 100
MIN_PATTERN_LENGTH = 10
OBSERVATION_MULT = 1

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    best_model_index = -1
    unguessed_fish = set(range(N_FISH))

    group_keys = set()
    group_key = None
    group_type = -1
    current_model = None

    def initialize_matrix(self, n, m):
        matrix = []
        for _ in range(n):
            row = [1 / m] * (m)

            for j in range(m):
                rand_number = random.gauss(100, 50)
                if rand_number < 0:
                    rand_number *= -1
                row[j] += rand_number
            
            norm = sum(row)
            for j in range(m):
                row[j] /= norm
                
            matrix.append(row)
        return matrix

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        random.seed(str(1))
        self.A = self.initialize_matrix(N_SPECIES, N_SPECIES)
        self.B = self.initialize_matrix(N_SPECIES, N_EMISSIONS)
        self.pi = self.initialize_matrix(1, N_SPECIES)[0]

        self.As = []
        self.Bs = []
        self.pies = []
        
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

    def get_next_element(self, source_set):
        try:
            return source_set.pop()
        except:
            return None

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
            self.groups = self.match_maximum(self.observations, MIN_PATTERN_LENGTH, True)
            print(self.groups)

            # unseen_fish = set(range(N_FISH))
            # for value in self.groups.values():
            #     self.training_set.append(self.observations[list(value)[0]])
            #     for i in value:
            #         unseen_fish.discard(i)

            # for i in unseen_fish:
            #     self.training_set.append(self.observations[i])
        
        if step >= THRESHOLD_STEP + 1:
            fish_index = -1

            if self.group_key is None:
                self.group_keys = set(self.groups.keys())
                self.group_key = self.get_next_element(self.group_keys)

            if not self.groups[self.group_key]:
                self.group_key = self.get_next_element(self.group_keys)
                self.group_type = -1
            
            if self.group_key is None:
                fish_index = self.get_next_element(self.unguessed_fish)

                if self.best_model_index != -1:
                    path = ut.viterbi(self.current_model[0], self.current_model[1], self.initialize_matrix(1, N_SPECIES)[0], self.observations[fish_index])
                    self.best_model_index = -1
                    
                    print(fish_index, max(path, key=path.count), path)
                    return fish_index, max(path, key=path.count)
            else:
                print('Group', self.group_key)
                fish_index = self.get_next_element(self.groups[self.group_key])
                while fish_index not in self.unguessed_fish:
                    fish_index = self.get_next_element(self.groups[self.group_key])
                if not fish_index:
                    print('No more fish in this group')
                    return None

                if self.group_type != -1:
                    return fish_index, self.group_type
                    
            try:
                more_observations = self.observations[fish_index] * OBSERVATION_MULT
                A, B, pi = ut.baum_welch(
                    self.initialize_matrix(N_SPECIES, N_SPECIES), self.initialize_matrix(N_SPECIES, N_EMISSIONS), self.initialize_matrix(1, N_SPECIES)[0], more_observations)
                self.current_model = (A, B, pi)
            except:
                print('Failed to do baum welch for fish', fish_index)
                return None

            path = ut.viterbi(self.current_model[0], self.current_model[1], self.current_model[2], self.observations[fish_index])
            print(fish_index, np.argmax(self.current_model[2]), max(path, key=path.count), path)

            return fish_index, max(path, key=path.count)
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
        if correct:
            self.best_model_index = fish_id
            self.group_type = true_type
        self.unguessed_fish.remove(fish_id)
