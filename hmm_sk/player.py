#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import math
import numpy as np
import utils as ut
import sys
from collections import namedtuple
from difflib import SequenceMatcher
import operator

THRESHOLD_STEP = 100
N_STATES = 2

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    models = dict()
    last_fish_type = -1
    last_fish_guessed = -1
    good_guess = False
    unguessed_fish = set(range(N_FISH))
    # counter = 0
    group_observations = dict()

    def initialize_matrix(self, n, m):
        matrix = []
        for _ in range(n):
            row = [1 / m] * (m)

            for j in range(m):
                rand_number = random.gauss(0, 100)
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

    def build_model(self, model_type):
        self.models[model_type] = ut.baum_welch(
            self.initialize_matrix(N_STATES, N_STATES), self.initialize_matrix(N_STATES, N_EMISSIONS),
            self.initialize_matrix(1, N_STATES)[0], self.group_observations[model_type])

    def improve_model(self, fish, model_type):
        # print("Improving model", model_type, "with fish", fish)
        self.group_observations[model_type] += self.observations[fish]
        self.build_model(model_type)

    def get_best_fish(self, model_type):
        
        log_probs = {}

        for fish in self.unguessed_fish:
            try: 
                alpha, ct= ut.alpha_pass(self.models[model_type][0], self.models[model_type][1],
                                        self.initialize_matrix(1, N_STATES)[0], self.observations[fish])
                for c in ct: 
                    c = math.log(c)
                log_probs[fish] = -sum(ct)
            except:
                pass

        if len(log_probs) > 0:                      
            log_probs = dict(
                sorted(log_probs.items(), key=operator.itemgetter(1), reverse=True))
            best_fish = list(log_probs.keys())[0]
        else: 
            return -1, float('-inf')
        return best_fish, log_probs[best_fish]

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
            return random.sample(self.unguessed_fish, 1)[0], random.randrange(N_SPECIES)

        if step >= THRESHOLD_STEP + 1:
            if self.last_fish_type not in self.models or not self.good_guess:
                self.improve_model(self.last_fish_guessed, self.last_fish_type)
            
            best_fish = -1
            fish_type = -1
            highest_prob = float('-inf')
            for model in self.models:
                fish, prob = self.get_best_fish(model)
                # print('best_fish for model ', model, 'is', fish, 'highest_prob',prob, file=sys.stderr)
                if prob > highest_prob:
                    best_fish = fish
                    fish_type = model
                    highest_prob = prob

            if(best_fish == -1):
                # print('No best Fish found')
                return random.sample(self.unguessed_fish, 1)[0], random.randrange(N_SPECIES)

            # print('Guess', best_fish, 'type', fish_type, file=sys.stderr)
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
        # print('Result', 'True type', true_type, correct, file=sys.stderr)
        self.last_fish_type = true_type
        self.last_fish_guessed = fish_id
        self.unguessed_fish.remove(fish_id)
        self.good_guess = correct
        # if not correct:
        #     self.counter += 1
        # print("there's ", len(self.unguessed_fish), "fishes left", file=sys.stderr)
