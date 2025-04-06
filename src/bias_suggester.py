import re

# Add/expand this dictionary over time
bias_dictionary = {
    "idiot": "individual",
    "stupid": "uninformed",
    "dumb": "misguided",
    "hate": "dislike",
    "ugly": "unpleasant",
    "retard": "person with a disability",
    "faggot": "person",
    "bitch": "person",
    "cunt": "individual",
    "mongoloid": "person",
    "twat": "individual",
    "nigger": "individual",
    "dyke": "person",
    # Add more as needed
}

def suggest_replacements(text):
    # Lowercase the text for matching
    modified_text = text

    for word, replacement in bias_dictionary.items():
        # regex to match whole words only, case-insensitive
        pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
        modified_text = pattern.sub(f"[{replacement}]", modified_text)

    return modified_text
