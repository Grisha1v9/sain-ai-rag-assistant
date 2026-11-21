import re

ATTACK_REGEX = re.compile(
    r'\b(доз[аыуе]|дозировк|мг\b|мл\b|диагноз|пропиш|корвалол|феназепам|снотворн)\b', 
    re.IGNORECASE
)

def is_attack(text: str) -> bool:
    return bool(ATTACK_REGEX.search(text))