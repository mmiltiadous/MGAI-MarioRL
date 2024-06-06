import random


def choose_random(choice1, choice2, p1):
    return choice1 if random.random() > p1 else choice2


def choose_random_element(elements):
    return elements[random.randrange(0, len(elements))]


def clamp_value(value, min_value, max_value):
    return max(min(value, max_value), min_value)
