"""
This module defines the decision-making functions for various bot opponents.

Each function represents a different "personality" or strategy that the RL agent
can be trained against. These strategies are rule-based and are used to create
a diverse and challenging training environment.
"""
import random

def suspect_play(player, current_rank, last_number_of_cards_played):
    if current_rank == "Open":
        return False
    
    if last_number_of_cards_played > 4:
        return True
    
    cards_of_rank_in_hand = sum(1 for card in player.hand if card.value == current_rank)
    total_known_cards_of_rank = last_number_of_cards_played + cards_of_rank_in_hand

    if total_known_cards_of_rank > 4:
        return True
    return False


def bot_strategy_80_20(player, current_rank, last_number_of_cards_played):
    """
    Implements a conservative bot that prefers to play truthfully.

    When responding to a play, this bot has an 80% chance to be honest. If it
    cannot play a truthful hand, it will either pass or challenge. When
    starting a round, it identifies its most common rank and has a 75%
    chance to play that rank truthfully.
    """

    if suspect_play(player, current_rank, last_number_of_cards_played):
        if random.randint(1, 20) != 1:
            return (0, [], current_rank)

    if current_rank != "Open":
        true_cards = [card for card in player.hand if card.value == current_rank]
        fake_cards = [card for card in player.hand if card.value != current_rank and card.value != "Joker"]
        jokers = [card for card in player.hand if card.value == "Joker"]

        if len(fake_cards) == 0:
            return (2, player.hand, current_rank)

        tell_truth = False
        if random.randint(1, 5) != 1:
            tell_truth = True

        if tell_truth:
            if len(true_cards) > 0:
                return (2, true_cards, current_rank)
            elif len(true_cards) == 0 and len(jokers) > 0:
                return (2, [jokers[0]], current_rank)
            else:
                if random.randint(1, 2) == 1:
                    return (1, [], current_rank)
                else:
                    return (0, [], current_rank)
        else:
            if len(fake_cards) == 1:
                if len(player.hand) == 1:
                    return (1, [], current_rank)
                return (2, fake_cards, current_rank)
            else:
                if random.randint(1, 2) == 1:
                    return (2, [fake_cards[0]], current_rank)
                else:
                    if len(player.hand) <= 2:
                        return (1, [], current_rank)
                    return (2, [fake_cards[0], fake_cards[1]], current_rank)
    else:
        hand_values_no_jokers = [card.value for card in player.hand if card.value != "Joker"]
        if not hand_values_no_jokers:
            most_commun_rank = "Ace"
        else:
            most_commun_rank = max(set(hand_values_no_jokers), key=hand_values_no_jokers.count)

        card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        value_to_index = {value: i for i, value in enumerate(card_values)}
        hand_vector = [0]*14
        for card in player.hand:
            if card.value in value_to_index:
                hand_vector[value_to_index[card.value]] += 1

        number_of_ranks = 0
        max_value = 0
        for num in hand_vector:
            if num > 0:
                number_of_ranks += 1
            if num > max_value:
                max_value = num

        new_announced_rank = most_commun_rank

        if number_of_ranks == 1:
            return (2, player.hand, new_announced_rank)

        if random.randint(1, 4) == 1:
            tell_truth = False
        else:
            tell_truth = True

        if tell_truth:
            play_cards = []
            for card in player.hand:
                if card.value == new_announced_rank:
                    play_cards.append(card)
            return (2, play_cards, new_announced_rank)
        else:
            num_of_cards_to_play = random.randint(1, int(max_value))
            play_cards = []
            for card in player.hand:
                if len(play_cards) == num_of_cards_to_play:
                    break
                if card.value != new_announced_rank and card.value != "Joker":
                    play_cards.append(card)

            if len(play_cards) == 0:
                return (2, player.hand, new_announced_rank)
            else:
                return (2, play_cards, new_announced_rank)


def bot_strategy_one_third(player, current_rank, last_number_of_cards_played):
    """
    Implements a highly unpredictable bot.

    When responding to a play, this bot chooses with equal 1/3 probability
    between challenging, attempting a truthful play, or attempting to lie.
    When starting a round, it finds its most common rank and has a 50/50
    chance to either play it honestly or lie.
    """

    if suspect_play(player, current_rank, last_number_of_cards_played):
        if random.randint(1, 20) != 1:
            return (0, [], current_rank)
    
    if current_rank != "Open":
        true_cards = [card for card in player.hand if card.value == current_rank]
        fake_cards = [card for card in player.hand if card.value != current_rank and card.value != "Joker"]
        jokers = [card for card in player.hand if card.value == "Joker"]

        if len(fake_cards) == 0:
            return (2, player.hand, current_rank)

        decide_action = random.randint(1, 3)

        if decide_action == 1:
            if len(true_cards) > 0:
                return (2, true_cards, current_rank)
            elif len(true_cards) == 0 and len(jokers) > 0:
                return (2, [jokers[0]], current_rank)
            else:
                if random.randint(1, 2) == 1:
                    return (1, [], current_rank)
                else:
                    return (0, [], current_rank)
        elif decide_action == 2:
            if len(fake_cards) == 1:
                if len(player.hand) == 1:
                    return (1, [], current_rank)
                return (2, fake_cards, current_rank)
            else:
                if random.randint(1, 2) == 1:
                    return (2, [fake_cards[0]], current_rank)
                else:
                    if len(player.hand) <= 2:
                        return (1, [], current_rank)
                    return (2, [fake_cards[0], fake_cards[1]], current_rank)
        elif decide_action == 3:
            return (0, [], current_rank)
    else:
        hand_values_no_jokers = [card.value for card in player.hand if card.value != "Joker"]
        if not hand_values_no_jokers:
            most_commun_rank = "Ace"
        else:
            most_commun_rank = max(set(hand_values_no_jokers), key=hand_values_no_jokers.count)

        card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        value_to_index = {value: i for i, value in enumerate(card_values)}
        hand_vector = [0]*14
        for card in player.hand:
            if card.value in value_to_index:
                hand_vector[value_to_index[card.value]] += 1

        number_of_ranks = 0
        max_value = 0
        for num in hand_vector:
            if num > 0:
                number_of_ranks += 1
            if num > max_value:
                max_value = num

        new_announced_rank = most_commun_rank

        if number_of_ranks == 1:
            return (2, player.hand, new_announced_rank)

        if random.randint(1, 2) == 1:
            tell_truth = False
        else:
            tell_truth = True

        if tell_truth:
            play_cards = []
            for card in player.hand:
                if card.value == new_announced_rank:
                    play_cards.append(card)
            return (2, play_cards, new_announced_rank)
        else:
            num_of_cards_to_play = random.randint(1, int(max_value))
            play_cards = []
            for card in player.hand:
                if len(play_cards) == num_of_cards_to_play:
                    break
                if card.value != new_announced_rank and card.value != "Joker":
                    play_cards.append(card)

            if len(play_cards) == 0:
                return (2, player.hand, new_announced_rank)
            else:
                return (2, play_cards, new_announced_rank)


def bot_strategy_100_0(player, current_rank, last_number_of_cards_played):
    """
    Implements a completely honest bot.

    This bot **never lies** and never doubts. When it does not have truth cards, it passes.
    """

    if suspect_play(player, current_rank, last_number_of_cards_played):
        if random.randint(1, 20) != 1:
            return (0, [], current_rank)


    if current_rank != "Open":
        true_cards = [card for card in player.hand if card.value == current_rank]
        fake_cards = [card for card in player.hand if card.value != current_rank and card.value != "Joker"]
        jokers = [card for card in player.hand if card.value == "Joker"]

        if len(fake_cards) == 0:
            return (2, player.hand, current_rank)
        
        if len(true_cards) > 0:
            return (2, true_cards, current_rank)
        else:
            return (1, [], current_rank)
    else:
        hand_values_no_jokers = [card.value for card in player.hand if card.value != "Joker"]
        if not hand_values_no_jokers:
            most_commun_rank = "Ace"
        else:
            most_commun_rank = max(set(hand_values_no_jokers), key=hand_values_no_jokers.count)

        card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        value_to_index = {value: i for i, value in enumerate(card_values)}
        hand_vector = [0]*14
        for card in player.hand:
            if card.value in value_to_index:
                hand_vector[value_to_index[card.value]] += 1

        number_of_ranks = 0
        max_value = 0
        for num in hand_vector:
            if num > 0:
                number_of_ranks += 1
            if num > max_value:
                max_value = num

        new_announced_rank = most_commun_rank

        if number_of_ranks == 1:
            return (2, player.hand, new_announced_rank)
        
        play_cards = []
        for card in player.hand:
            if card.value == new_announced_rank:
                play_cards.append(card)
        return (2, play_cards, new_announced_rank)


def bot_strategy_60_40(player, current_rank, last_number_of_cards_played):
    """
    Implements a balanced bot strategy.

    When responding to a play, this bot has a 60% chance to be truthful and
    a 40% chance to lie. When starting a new round, it also has a 60% chance
    to start with a truthful play of its most common rank.
    """

    if suspect_play(player, current_rank, last_number_of_cards_played):
        if random.randint(1, 20) != 1:
            return (0, [], current_rank)

    if current_rank != "Open":
        true_cards = [card for card in player.hand if card.value == current_rank]
        fake_cards = [card for card in player.hand if card.value != current_rank and card.value != "Joker"]
        jokers = [card for card in player.hand if card.value == "Joker"]

        if len(fake_cards) == 0:
            return (2, player.hand, current_rank)

        tell_truth = False
        if random.randint(1, 5) > 2:
            tell_truth = True

        if tell_truth:
            if len(true_cards) > 0:
                return (2, true_cards, current_rank)
            elif len(true_cards) == 0 and len(jokers) > 0:
                return (2, [jokers[0]], current_rank)
            else:
                if random.randint(1, 2) == 1:
                    return (1, [], current_rank)
                else:
                    return (0, [], current_rank)
        else:
            if len(fake_cards) == 1:
                if len(player.hand) == 1:
                    return (1, [], current_rank)
                return (2, fake_cards, current_rank)
            else:
                if random.randint(1, 2) == 1:
                    return (2, [fake_cards[0]], current_rank)
                else:
                    if len(player.hand) <= 2:
                        return (1, [], current_rank)
                    return (2, [fake_cards[0], fake_cards[1]], current_rank)
    else:
        hand_values_no_jokers = [card.value for card in player.hand if card.value != "Joker"]
        if not hand_values_no_jokers:
            most_commun_rank = "Ace"
        else:
            most_commun_rank = max(set(hand_values_no_jokers), key=hand_values_no_jokers.count)

        card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        value_to_index = {value: i for i, value in enumerate(card_values)}
        hand_vector = [0]*14
        for card in player.hand:
            if card.value in value_to_index:
                hand_vector[value_to_index[card.value]] += 1

        number_of_ranks = 0
        max_value = 0
        for num in hand_vector:
            if num > 0:
                number_of_ranks += 1
            if num > max_value:
                max_value = num

        new_announced_rank = most_commun_rank

        if number_of_ranks == 1:
            return (2, player.hand, new_announced_rank)

        if random.randint(1, 5) < 3:
            tell_truth = False
        else:
            tell_truth = True

        if tell_truth:
            play_cards = []
            for card in player.hand:
                if card.value == new_announced_rank:
                    play_cards.append(card)
            return (2, play_cards, new_announced_rank)
        else:
            num_of_cards_to_play = random.randint(1, int(max_value))
            play_cards = []
            for card in player.hand:
                if len(play_cards) == num_of_cards_to_play:
                    break
                if card.value != new_announced_rank and card.value != "Joker":
                    play_cards.append(card)

            if len(play_cards) == 0:
                return (2, player.hand, new_announced_rank)
            else:
                return (2, play_cards, new_announced_rank)
            

def bot_strategy_challenger(player, current_rank, last_number_of_cards_played):
    """
    Implements a challenger and honest bot

    This bot **never lies**. But if the other player plays, it will always challenge.
    """

    if current_rank != "Open":
        return [0, [], current_rank]
    else:
        hand_values_no_jokers = [card.value for card in player.hand if card.value != "Joker"]
        if not hand_values_no_jokers:
            most_commun_rank = "Ace"
        else:
            most_commun_rank = max(set(hand_values_no_jokers), key=hand_values_no_jokers.count)

        card_values = ["Joker", "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
        value_to_index = {value: i for i, value in enumerate(card_values)}
        hand_vector = [0]*14
        for card in player.hand:
            if card.value in value_to_index:
                hand_vector[value_to_index[card.value]] += 1

        number_of_ranks = 0
        max_value = 0
        for num in hand_vector:
            if num > 0:
                number_of_ranks += 1
            if num > max_value:
                max_value = num

        new_announced_rank = most_commun_rank

        if number_of_ranks == 1:
            return (2, player.hand, new_announced_rank)
        
        play_cards = []
        for card in player.hand:
            if card.value == new_announced_rank:
                play_cards.append(card)
        return (2, play_cards, new_announced_rank)