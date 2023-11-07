import os
import language


# =====================================BASIC LANGUAGE TESTING, REGULAR PARADIGMS=================+======================
# Use the same language to make some ungrammatical conjugations
# Temporary, probably will get rid of this soon
# The
def make_wrong_sentences_basic():
    # Load the original json file
    mylang = language.load_language("Languages/mylang")

    # Forcefully restart the agreements to cycle, set the new inflection paradigms
    mylang.inflection_paradigms = []
    mylang.set_inflection_paradigms([
        ["verb", {
            ("sg", "1st"): "-me",
            ("sg", "2nd"): "-ju",
            ("sg", "3rd"): "-si",
            ("pl", "1st"): "-we",
            ("pl", "2nd"): "-jal",
            ("pl", "3rd"): "-dej"
        }],
        ["noun", {
            "sg": "-",
            "pl": "-ol"
        }]
    ])

    # Make ungrammatical sentences
    generate_and_save_sentences(wronglang, "wronglang", 100, "test_same_dist_ungrammatical")

    # Make ungrammatical sentences with verb roots never seen before
    # Generate some sentences with never seen before verbs, we know these below aren't in wronglang
    new_verb_lexemes = ["testo", "esamen", "falso", "finto", "fake", "niente", "nada", "nothin", "palabra", "wordo"]
    new_verbs = {
        "verb": [(verb, "new") for verb in new_verb_lexemes]
    }
    generate_and_save_sentences(wronglang, "wronglang", 100, "test_diff_dist_ungrammatical", new_verbs)


# Main method run
def main_basic():
    # Create/load a base language
    mylang = create_language_base()

    # Set the inflection paradigms
    mylang.set_inflection_paradigms([
        ["verb", {
            ("sg", "1st"): "-me",
            ("sg", "2nd"): "-ju",
            ("sg", "3rd"): "-si",
            ("pl", "1st"): "-we",
            ("pl", "2nd"): "-jal",
            ("pl", "3rd"): "-dej"
        }],
        ["noun", {
            "sg": "-",
            "pl": "-ol"
        }]
    ])

    # Generate 100 nouns specific to this language
    for amount, noun_property in [(1, "1st"), (1, "2nd"), (30, "3rd"), (1, "2nd"), (600, "3rd")]:
        mylang.generate_words(num_words=amount, part_of_speech="noun", paradigm=noun_property)
    mylang.generate_words(num_words=700, part_of_speech="verb", paradigm="verb1")

    # Save the language
    mylang.dump_language("Languages/mylang")

    # Generate some training sentences and save them
    sentences = generate_and_save_sentences(mylang, "mylang", 150000, "train")
    print(f'There are {len(set(sentences))} unique sentences.')

    # Generate a test set with sentences never seen before
    new_sentences = []
    while len(new_sentences) < 100:
        new_sentence = mylang.generate_sentences(1)
        if new_sentence not in sentences:
            new_sentences += new_sentence
    print(new_sentences)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages/mylang/100_test_same_distribution_sentences.txt"))

    # Generate some sentences with never seen before verbs
    new_verb_lexemes = ["testo", "esamen", "falso", "finto", "fake", "niente", "nada", "nothin", "palabra", "wordo"]
    new_verbs = {
        "verb": [(verb, "new") for verb in new_verb_lexemes]
    }

    # For now, just make sure these words aren't in the training set
    preexisting_words = set(new_verbs["verb"]) & mylang.word_set
    print(preexisting_words)
    assert len(preexisting_words) == 0

    # Generate 100 test sentences with the new verbs
    generate_and_save_sentences(mylang, "mylang", 100, "test_diff_distribution", new_verbs)


# =====================================BASIC LANGUAGE TESTING, VERB CLASSES======================+======================
# Makes verbs according to two verb classes of near equal frequency
def main_verb_classes():
    # Create/load a base language
    mylang = create_language_base()

    # Set the inflection paradigms
    mylang.set_inflection_paradigms([
        ["verb", {
            ("sg", "1st", "p1"): "-me",
            ("sg", "2nd", "p1"): "-ju",
            ("sg", "3rd", "p1"): "-si",
            ("pl", "1st", "p1"): "-we",
            ("pl", "2nd", "p1"): "-jal",
            ("pl", "3rd", "p1"): "-dej",
            ("sg", "1st", "p2"): "-me",
            ("sg", "2nd", "p2"): "-ju",
            ("sg", "3rd", "p2"): "-si",
            ("pl", "1st", "p2"): "-we",
            ("pl", "2nd", "p2"): "-jal",
            ("pl", "3rd", "p2"): "-dej"
        }],
        ["noun", {
            "sg": "-",
            "pl": "-ol"
        }]
    ])

    # Generate 100 nouns specific to this language
    for amount, noun_property in [(1, "1st"), (1, "2nd"), (30, "3rd"), (1, "2nd"), (600, "3rd")]:
        mylang.generate_words(num_words=amount, part_of_speech="noun", paradigm=noun_property)
    # Generate 350 words from each paradigm with approximately equal probability
    for _ in range(350):
        mylang.generate_words(num_words=1, part_of_speech="verb", paradigm="p1")
        mylang.generate_words(num_words=1, part_of_speech="verb", paradigm="p2")

    # Save the language
    mylang.dump_language("Languages/simple_paradigms")

    # Generate some training sentences and save them
    sentences = generate_and_save_sentences(mylang, "simple_paradigms", 150000, "train")
    print(f'There are {len(set(sentences))} unique sentences.')

    # Generate a test set with sentences never seen before
    new_sentences = []
    while len(new_sentences) < 100:
        new_sentence = mylang.generate_sentences(1)
        if new_sentence not in sentences:
            new_sentences += new_sentence
    print(new_sentences)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join(
                                "Languages/simple_paradigms/100_test_same_distribution_sentences.txt"))

    # Generate some sentences with never seen before verbs
    new_verb_lexemes = ["testo", "esamen", "falso", "finto", "fake", "niente", "nada", "nothin", "palabra", "wordo"]
    new_verbs = {
        "verb": [(verb, "new.p1" if index % 2 else "new.p2") for index, verb in enumerate(new_verb_lexemes)]
    }

    # For now, just make sure these words aren't in the training set
    preexisting_words = set(new_verbs["verb"]) & mylang.word_set
    print(preexisting_words)
    assert len(preexisting_words) == 0

    # Generate 100 test sentences with the new verbs
    generate_and_save_sentences(mylang, "simple_paradigms", 100, "test_diff_distribution", new_verbs)


# =====================================GENERALLY HELPFUL METHODS========================================================

def generate_and_save_sentences(lang, language_name, num_sentences, sentence_prefix, required_words=None):
    # Generate some sentences
    sentences = lang.generate_sentences(num_sentences, required_words)

    # Create the directory if it does not exist
    directory_path = os.path.join("Languages", language_name)
    os.makedirs(directory_path, exist_ok=True)
    # Save them
    language.save_sentences(sentences=sentences,
                            filepath=os.path.join(directory_path, f"{num_sentences}_{sentence_prefix}_sentences.txt"))

    # Return the sentences you generates
    return sentences


# Creates a test language
def create_language_base():
    # Create a language
    mylang = language.Language()

    # Set the phonology of the language
    phonemes = {
        "C": ["p", "b", "t", "d", "k", "g", "m", "n", "f", "v", "s", "z", "h", "j", "w", "r", "l"],
        "V": ["a", "e", "i", "o", "u"]
    }
    mylang.set_phonemes(phonemes=phonemes)

    # Set the syllables of the language
    syllables = ["CVC", "CV", "VC"]
    mylang.set_syllables(syllables=syllables)

    # Set the syllable lambda
    mylang.set_syllable_lambda(0.8)

    # Set the parts of speech of the language
    parts_of_speech = ["noun", "verb"]
    mylang.set_parts_of_speech(parts_of_speech=parts_of_speech)

    # Set the generation rules
    mylang.set_generation_rules({
        "S": [["sN", "VP"], 1],
        "VP": [["verb", "N"], 0.7, ["verb"], 0.3],
    })

    # Set independent probabilistic rules, e.g. pluralization, past tense
    mylang.set_unconditioned_rules({
        "sN": [["N"], "nom", 1],
        "N": [["noun"], "sg", 0.8, "pl", 0.2]
    })

    # Set the agreement rules
    mylang.set_agreement_rules({
        "verb": [["nom"], [["sg", "pl"], ["1st", "2nd", "3rd"]]]
    })

    # Return the language
    return mylang


if __name__ == "__main__":
    # main_basic()
    # make_wrong_sentences_basic()
    main_verb_classes()
