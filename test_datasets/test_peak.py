test_cases = [
    (["BOS", 1, 2, 3],       ["BOS", False, False, True]),
    (["BOS", 0, 0, 0],       ["BOS", False, False, False]),
    (["BOS", 4, 3, 2, 1],    ["BOS", True, False, False, False]),
    (["BOS", 1, 1, 1, 1],    ["BOS", False, False, False, False]),
    (["BOS", 9, 9, 9],       ["BOS", False, False, False]),
    (["BOS", 1, 2, 3, 4, 5], ["BOS", False, False, False, False, True]),
    (["BOS", 2, 2, 3, 2],    ["BOS", False, False, True, False]),
    (["BOS", 1, 3, 1, 3, 1], ["BOS", False, True, False, True, False]),
    (["BOS", 0, 1, 0, 1, 0], ["BOS", False, True, False, True, False]),
    (["BOS", 2, 4, 2, 4, 2], ["BOS", False, True, False, True, False]),
    (["BOS", 1, 5, 1, 5, 1], ["BOS", False, True, False, True, False]),
    (["BOS", 9, 8, 7, 6, 5], ["BOS", True, False, False, False, False]),
    (["BOS", 1, 2, 1, 2, 1], ["BOS", False, True, False, True, False]),
    (["BOS", 0, 2, 0, 2, 0], ["BOS", False, True, False, True, False]),
    (["BOS", 2, 3, 4, 3, 2], ["BOS", False, False, True, False, False]),
    (["BOS", 1, 4, 1, 4, 1], ["BOS", False, True, False, True, False]),
    (["BOS", 0, 1, 2, 1, 0], ["BOS", False, False, True, False, False]),
    (["BOS", 1, 3, 5, 3, 1], ["BOS", False, False, True, False, False]),
    (["BOS", 2, 2, 2, 2, 2], ["BOS", False, False, False, False, False]),
    (["BOS", 3, 5, 7, 5, 3], ["BOS", False, False, True, False, False]),

    (["BOS", 0, 1, 0, 1, 0], ["BOS", False, True, False, True, False]),
    (["BOS", 2, 4, 2, 4, 2], ["BOS", False, True, False, True, False]),
    (["BOS", 1, 3, 1, 3, 1], ["BOS", False, True, False, True, False]),
    (["BOS", 0, 3, 0, 3, 0], ["BOS", False, True, False, True, False]),
    (["BOS", 1, 4, 1, 4, 1], ["BOS", False, True, False, True, False]),
    (["BOS", 2, 5, 2, 5, 2], ["BOS", False, True, False, True, False]),
    (["BOS", 3, 6, 3, 6, 3], ["BOS", False, True, False, True, False]),
    (["BOS", 4, 7, 4, 7, 4], ["BOS", False, True, False, True, False]),
    (["BOS", 5, 8, 5, 8, 5], ["BOS", False, True, False, True, False]),
    (["BOS", 6, 9, 6, 9, 6], ["BOS", False, True, False, True, False]),

    # failing test cases were fixed! 
    (["BOS", 3, 2, 3, 2, 3], ["BOS", True, False, True, False, True]),
    (["BOS", 4, 3, 4, 3, 4], ["BOS", True, False, True, False, True]),
    (["BOS", 5, 4, 5, 4, 5], ["BOS", True, False, True, False, True]),
    (["BOS", 6, 5, 6, 5, 6], ["BOS", True, False, True, False, True]),
    (["BOS", 8, 0, 2],       ["BOS", True, False, True]),
    (["BOS", 3, 5, 2, 4],    ["BOS", False, True, False, True]),
    (["BOS", 7, 5, 7, 5],    ["BOS", True, False, True, False]),
    (["BOS", 3, 1, 3, 1, 3], ["BOS", True, False, True, False, True]),
    (["BOS", 5, 3, 5, 3, 5], ["BOS", True, False, True, False, True]),
    (["BOS", 1, 0, 1, 0, 1], ["BOS", True, False, True, False, True]),
    (["BOS", 5, 1, 5, 1, 5], ["BOS", True, False, True, False, True]),
    (["BOS", 6, 2, 6, 2, 6], ["BOS", True, False, True, False, True]),
    (["BOS", 7, 0, 7, 0, 7], ["BOS", True, False, True, False, True]),
    (["BOS", 8, 1, 8, 1, 8], ["BOS", True, False, True, False, True]),
    (["BOS", 9, 2, 9, 2, 9], ["BOS", True, False, True, False, True]),
    (["BOS", 1, 0, 1, 0, 1], ["BOS", True, False, True, False, True]),
    (["BOS", 2, 1, 2, 1, 2], ["BOS", True, False, True, False, True]),

    # single numbers or boundary elements lack both neighbors -> ambiguous whether they are peaks
    # due to the missing right neighbor, we get False instead of True 
    (["BOS", 0],             ["BOS", False]),
    (["BOS", 5],             ["BOS", False]),
]

def get_peak_test_cases(): # so that main_peak.py can obtain the test cases from here
    return test_cases

