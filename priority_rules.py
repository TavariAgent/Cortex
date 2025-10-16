# Centralised operator precedence for all engines
PRIORITY = {
    '^': 4,
    '*': 3, '/': 3,
    '+': 2, '-': 2,
    'sum': 1, 'limit': 1,
    'other': 0,
}

def precedence_of(token: str) -> int:
    return PRIORITY.get(token, 0)