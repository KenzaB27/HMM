#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np
import utils as ut
import sys
from collections import namedtuple
from difflib import SequenceMatcher
import operator

THRESHOLD_STEP = 80
MIN_PATTERN_LENGTH = 10
N_STATES = N_SPECIES
# OBSERVATION_MULT = 1


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    models = dict()
    last_fish_type = -1
    last_fish_guessed = -1
    good_guess = False
    unguessed_fish = set(range(N_FISH))
    group_observations = dict()
    improvements = dict()

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
        self.observations = [[] for _ in range(N_FISH)]

        self.group_observations = {i: [] for i in range(N_SPECIES)}

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


    def build_model(self, model_type):
        if model_type in self.models:
            self.models[model_type] = ut.baum_welch(
                self.models[model_type][0],
                self.models[model_type][1],
                self.initialize_matrix(1, N_STATES)[0],
                self.group_observations[model_type]
            )
        else:
            self.models[model_type] = ut.baum_welch(
                self.initialize_matrix(N_STATES, N_STATES),
                self.initialize_matrix(N_STATES, N_EMISSIONS),
                self.initialize_matrix(1, N_STATES)[0],
                self.group_observations[model_type]
            )

    def improve_model(self, observations, model_type):
        # self.group_observations[model_type] = observations
        # option 2
        if not self.group_observations[model_type]:
            self.group_observations[model_type] = observations
        else:
            l = len(self.group_observations[model_type])
            self.group_observations[model_type] = self.group_observations[model_type][:10] + observations
            
        self.build_model(model_type)

    def get_best_fish(self, model_type):
        probs = {}

        for fish in self.unguessed_fish:
            alpha = ut.alpha_pass_no_scaling(self.models[model_type][0], self.models[model_type][1], self.initialize_matrix(1, N_STATES)[0], self.observations[fish])
            prob = sum([ alpha[t][i] for t in range(len(self.observations[fish])) for i in range(N_STATES) ])
            probs[fish] = prob

        probs = dict(sorted(probs.items(), key=operator.itemgetter(1), reverse=True))
        best_fish = list(probs.keys())[0]
        return best_fish, probs[best_fish]

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        for i in range(N_FISH):
            self.observations[i].append(observations[i])

        if step == THRESHOLD_STEP:
            return 0, random.randrange(N_SPECIES)
        
        if step >= THRESHOLD_STEP + 1:
            # if not self.models or not self.good_guess:
            if self.last_fish_type not in self.improvements:
                self.improvements[self.last_fish_type] = 0
            if self.improvements[self.last_fish_type] < 4:
                self.improve_model(self.observations[self.last_fish_guessed], self.last_fish_type)
                self.improvements[self.last_fish_type] += 1
            
            best_fish = -1
            fish_type = -1
            highest_prob = 0.0

            for model in self.models:
                fish, prob = self.get_best_fish(model)
                if prob > highest_prob:
                    best_fish = fish
                    fish_type = model
                    highest_prob = prob

            return best_fish, fish_type
            
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
        self.last_fish_type = true_type
        self.last_fish_guessed = fish_id
        self.unguessed_fish.remove(fish_id)
        self.good_guess = correct
        print(len(self.unguessed_fish), file=sys.stderr)