import random
from typing import List, Tuple
from nltk.corpus import wordnet
from transformers import AutoTokenizer

class NLIAugmentor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLIAugmentor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

    def get_synonyms(self, word: str) -> List[str]:
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        return list(set(synonyms))

    def replace_with_synonyms(self, text: str, num_replacements: int = 2) -> str:
        words = text.split()
        if len(words) <= num_replacements:
            return text

        # Get indices of words that can be replaced
        replaceable_indices = [i for i, word in enumerate(words)
                             if len(self.get_synonyms(word)) > 0]

        if len(replaceable_indices) < num_replacements:
            return text

        # Randomly select indices to replace
        selected_indices = random.sample(replaceable_indices, num_replacements)

        # Replace selected words with synonyms
        for idx in selected_indices:
            synonyms = self.get_synonyms(words[idx])
            if synonyms:
                words[idx] = random.choice(synonyms)

        return ' '.join(words)

    def augment(self, premise: str, hypothesis: str, label: int) -> List[Tuple[str, str, int]]:
        augmented_pairs = [(premise, hypothesis, label)]  # Original pair

        # Create variations with premise modifications
        aug_premise = self.replace_with_synonyms(premise)
        if aug_premise != premise:
            augmented_pairs.append((aug_premise, hypothesis, label))

        # Create variations with hypothesis modifications
        aug_hypothesis = self.replace_with_synonyms(hypothesis)
        if aug_hypothesis != hypothesis:
            augmented_pairs.append((premise, aug_hypothesis, label))

        return augmented_pairs 