import os
import language


# =====================================BASIC LANGUAGE TESTING, NON SUPPLETIVE ALLOMORPHY============+===================
# Use the same language to make some ungrammatical conjugations
# Temporary, probably will get rid of this soon
def make_test_sentences_non_suppletive_allomorphy(train_sentences, language_name="non_suppletive_allomorphy"):
    # Load the original json file
    mylang = language.load_language(os.path.join("Languages", language_name))

    # Generate 1000 sentences from our four test distributions:
    # 1. Correct grammar, same distribution of words
    # 2. Incorrect grammar, same distribution of words
    # 3. Correct grammar, different distribution of words
    # 4. Incorrect grammar, different distribution of words
    num_sentences = 1000

    # 1. We start with the sentences from the same distribution
    # Generate a test set with sentences never seen before
    new_sentences = []
    new_agreed_lexeme_sequences = []
    while len(new_sentences) < num_sentences:
        new_sentence, agreed_lexeme_sequence = mylang.generate_sentences(1)
        # New sentence is a list containing just one element
        if new_sentence[0] not in train_sentences:
            new_sentences += new_sentence
            new_agreed_lexeme_sequences += agreed_lexeme_sequence
    print(new_sentences)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test1_{num_sentences}_correct_same.txt"))

    # 2. Now we take these agreed_lexeme_sequences and generate surface forms with our own rules
    # Here are the new paradigms
    incorrect_paradigms = [
        ["verb", {
            ("sg", "1st", "/C_"): "-me",
            ("sg", "2nd", "/C_"): "-ju",
            ("sg", "3rd", "/C_"): "-si",
            ("pl", "1st", "/C_"): "-we",
            ("pl", "2nd", "/C_"): "-jal",
            ("pl", "3rd", "/C_"): "-dej",
            ("sg", "1st", "/V_"): "-ame",
            ("sg", "2nd", "/V_"): "-aju",
            ("sg", "3rd", "/V_"): "-asi",
            ("pl", "1st", "/V_"): "-awe",
            ("pl", "2nd", "/V_"): "-ajal",
            ("pl", "3rd", "/V_"): "-adej"
        }],
        ["noun", {
            "sg": "-",
            "pl": "-ol"
        }]
    ]
    # Here we generate these sentences by simply inflecting the previous sequences with the correct paradigms
    new_sentences = language.inflect(new_agreed_lexeme_sequences, incorrect_paradigms, mylang.phonemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test2_{num_sentences}_incorrect_same.txt"))

    # 3. Generate some sentences with never seen before verbs
    new_verbs_p1 = mylang.generate_words(5, "verb", "/C_")
    new_verbs_p2 = mylang.generate_words(5, "verb", "/V_")
    new_verb_lexemes = {"verb": [*new_verbs_p1, *new_verbs_p2]}
    # This should never raise an error since generated words are guaranteed to be unique
    assert len(set(new_verb_lexemes["verb"]) & mylang.word_set) == 0
    # Generate these test sentences with the new verbs
    new_sentences, new_agreed_lexeme_sequences = mylang.generate_sentences(num_sentences,
                                                                           required_words=new_verb_lexemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test3_{num_sentences}_correct_diff.txt"))

    # 4. Generate some ungrammatical sentences with never seen before verbs
    # Here we generate these sentences by simply inflecting the previous sequences with the incorrect paradigms
    new_sentences = language.inflect(new_agreed_lexeme_sequences, incorrect_paradigms, mylang.phonemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test4_{num_sentences}_incorrect_diff.txt"))


# Verbs conjugate regularly, but add an epenthetic vowel added based on whether the suffix causes CC
def main_non_suppletive_allomorphy(language_name="non_suppletive_allomorphy"):
    # Create/load a base language
    mylang = create_language_base()

    # Set the inflection paradigms
    mylang.set_inflection_paradigms([
        ["verb", {
            ("sg", "1st", "/C_"): "-ame",
            ("sg", "2nd", "/C_"): "-aju",
            ("sg", "3rd", "/C_"): "-asi",
            ("pl", "1st", "/C_"): "-awe",
            ("pl", "2nd", "/C_"): "-ajal",
            ("pl", "3rd", "/C_"): "-adej",
            ("sg", "1st", "/V_"): "-me",
            ("sg", "2nd", "/V_"): "-ju",
            ("sg", "3rd", "/V_"): "-si",
            ("pl", "1st", "/V_"): "-we",
            ("pl", "2nd", "/V_"): "-jal",
            ("pl", "3rd", "/V_"): "-dej"
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
    for _ in range(750):
        mylang.generate_words(num_words=1, part_of_speech="verb", paradigm="p0")

    # Save the language
    mylang.dump_language(os.path.join("Languages", language_name))

    # Generate some training sentences and save them
    sentences = generate_and_save_sentences(mylang, language_name, 150000, "train")
    print(f'There are {len(set(sentences))} unique sentences.')

    # Return the training sentences
    return sentences


# =====================================BASIC LANGUAGE TESTING, REGULAR PARADIGMS=================+======================
# Use the same language to make some ungrammatical conjugations
# Temporary, probably will get rid of this soon
def make_test_sentences_basic(train_sentences, language_name="one_regular_paradigm"):
    # Load the original json file
    mylang = language.load_language(os.path.join("Languages", language_name))

    # Generate 1000 sentences from our four test distributions:
    # 1. Correct grammar, same distribution of words
    # 2. Incorrect grammar, same distribution of words
    # 3. Correct grammar, different distribution of words
    # 4. Incorrect grammar, different distribution of words
    num_sentences = 1000

    # 1. We start with the sentences from the same distribution
    # Generate a test set with sentences never seen before
    new_sentences = []
    new_agreed_lexeme_sequences = []
    while len(new_sentences) < num_sentences:
        new_sentence, agreed_lexeme_sequence = mylang.generate_sentences(1)
        # New sentence is a list containing just one element
        if new_sentence[0] not in train_sentences:
            new_sentences += new_sentence
            new_agreed_lexeme_sequences += agreed_lexeme_sequence
    print(new_sentences)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test1_{num_sentences}_correct_same.txt"))

    # 2. Now we take these agreed_lexeme_sequences and generate surface forms with our own rules
    # Here are the new paradigms
    incorrect_paradigms = [
        ["verb", {
            ("sg", "1st"): "-dej",
            ("sg", "2nd"): "-me",
            ("sg", "3rd"): "-ju",
            ("pl", "1st"): "-si",
            ("pl", "2nd"): "-we",
            ("pl", "3rd"): "-jal"
        }],
        ["noun", {
            "sg": "-",
            "pl": "-ol"
        }]
    ]
    # Here we generate these sentences by simply inflecting the previous sequences with the correct paradigms
    new_sentences = language.inflect(new_agreed_lexeme_sequences, incorrect_paradigms, mylang.phonemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test2_{num_sentences}_incorrect_same.txt"))

    # 3. Generate some sentences with never seen before verbs
    new_verb_lexemes = {"verb": mylang.generate_words(10, "verb", "new")}
    # This should never raise an error since generated words are guaranteed to be unique
    assert len(set(new_verb_lexemes["verb"]) & mylang.word_set) == 0
    # Generate these test sentences with the new verbs
    new_sentences, new_agreed_lexeme_sequences = mylang.generate_sentences(num_sentences,
                                                                           required_words=new_verb_lexemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test3_{num_sentences}_correct_diff.txt"))

    # 4. Generate some ungrammatical sentences with never seen before verbs
    # Here we generate these sentences by simply inflecting the previous sequences with the incorrect paradigms
    new_sentences = language.inflect(new_agreed_lexeme_sequences, incorrect_paradigms, mylang.phonemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test4_{num_sentences}_incorrect_diff.txt"))


# Main method run
def main_basic(language_name="one_regular_paradigm"):
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
    mylang.dump_language(os.path.join("Languages", language_name))

    # Generate some training sentences and save them
    sentences = generate_and_save_sentences(mylang, language_name, 150000, "train")
    print(f'There are {len(set(sentences))} unique sentences.')

    # Return the train sentences for us to compare with
    return sentences


# =====================================BASIC LANGUAGE TESTING, VERB CLASSES======================+======================
# Use the same language to make some ungrammatical conjugations
def make_test_sentences_classes(train_sentences, language_name="two_regular_classes"):
    # Load the original json file
    mylang = language.load_language(os.path.join("Languages", language_name))

    # Generate 1000 sentences from our four test distributions:
    # 1. Correct grammar, same distribution of words
    # 2. Incorrect grammar, same distribution of words
    # 3. Correct grammar, different distribution of words
    # 4. Incorrect grammar, different distribution of words
    num_sentences = 1000

    # 1. We start with the sentences from the same distribution
    # Generate a test set with sentences never seen before
    new_sentences = []
    new_agreed_lexeme_sequences = []
    while len(new_sentences) < num_sentences:
        new_sentence, agreed_lexeme_sequence = mylang.generate_sentences(1)
        # New sentence is a list containing just one element
        if new_sentence[0] not in train_sentences:
            new_sentences += new_sentence
            new_agreed_lexeme_sequences += agreed_lexeme_sequence
    print(new_sentences)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test1_{num_sentences}_correct_same.txt"))

    # 2. Now we take these agreed_lexeme_sequences and generate surface forms with our own rules
    # Here are the new paradigms
    # We just flipped which were p1 and p2
    incorrect_paradigms = [
        ["verb", {
            ("sg", "1st", "p2"): "-me",
            ("sg", "2nd", "p2"): "-ju",
            ("sg", "3rd", "p2"): "-si",
            ("pl", "1st", "p2"): "-we",
            ("pl", "2nd", "p2"): "-jal",
            ("pl", "3rd", "p2"): "-dej",
            ("sg", "1st", "p1"): "-jo",
            ("sg", "2nd", "p1"): "-tu",
            ("sg", "3rd", "p1"): "-essi",
            ("pl", "1st", "p1"): "-noj",
            ("pl", "2nd", "p1"): "-voj",
            ("pl", "3rd", "p1"): "-loro"
        }],
        ["noun", {
            "sg": "-",
            "pl": "-ol"
        }]
    ]
    # Here we generate these sentences by simply inflecting the previous sequences with the correct paradigms
    new_sentences = language.inflect(new_agreed_lexeme_sequences, incorrect_paradigms, mylang.phonemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test2_{num_sentences}_incorrect_same.txt"))

    # 3. Generate some sentences with never seen before verbs
    new_verbs_p1 = mylang.generate_words(5, "verb", "new.p1")
    new_verbs_p2 = mylang.generate_words(5, "verb", "new.p2")
    new_verb_lexemes = {"verb": [*new_verbs_p1, *new_verbs_p2]}
    # This should never raise an error since generated words are guaranteed to be unique
    assert len(set(new_verb_lexemes["verb"]) & mylang.word_set) == 0
    # Generate these test sentences with the new verbs
    new_sentences, new_agreed_lexeme_sequences = mylang.generate_sentences(num_sentences,
                                                                           required_words=new_verb_lexemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test3_{num_sentences}_correct_diff.txt"))

    # 4. Generate some ungrammatical sentences with never seen before verbs
    # Here we generate these sentences by simply inflecting the previous sequences with the incorrect paradigms
    new_sentences = language.inflect(new_agreed_lexeme_sequences, incorrect_paradigms, mylang.phonemes)
    # Save them
    language.save_sentences(sentences=new_sentences,
                            filepath=os.path.join("Languages",
                                                  language_name,
                                                  f"test4_{num_sentences}_incorrect_diff.txt"))


# Makes verbs according to two verb classes of near equal frequency
def main_verb_classes(language_name="two_regular_classes"):
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
            ("sg", "1st", "p2"): "-jo",
            ("sg", "2nd", "p2"): "-tu",
            ("sg", "3rd", "p2"): "-essi",
            ("pl", "1st", "p2"): "-noj",
            ("pl", "2nd", "p2"): "-voj",
            ("pl", "3rd", "p2"): "-loro"
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
    mylang.dump_language(os.path.join("Languages", language_name))

    # Generate some training sentences and save them
    sentences = generate_and_save_sentences(mylang, language_name, 150000, "train")
    print(f'There are {len(set(sentences))} unique sentences.')

    # Return the training sentences
    return sentences


# =====================================GENERALLY HELPFUL METHODS========================================================

def generate_and_save_sentences(lang, language_name, num_sentences, sentence_prefix, required_words=None):
    # Generate some sentences
    sentences, _ = lang.generate_sentences(num_sentences, required_words)

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
    parts_of_speech = ["noun", "verb", "adj", "prep"]
    mylang.set_parts_of_speech(parts_of_speech=parts_of_speech)

    # Set the generation rules
    # Adj N (Prep Adj N) V Adj O (Prep Adj N)
    mylang.set_generation_rules({
        "S": [["sNP", "VP"], 1],  # Sentences generate subject NPs and VPs
        "VP": [["verb", "NP"], 0.7, ["verb"], 0.3],  # VPs generate verbs (and object NPs)
        "NP": [["det", "NOM"], 1],  # All NPs require a determiner
        "NOM": [["adj", "NoAdjNOM"], 0.35, ["NoAdjNOM"], 0.65],  # NPs may take adjectives before the rest
        "NoAdjNP": [["N", "PP*nom"], 0.2, ["N"], 0.8],  # NoAdjNPs become nouns, or nouns with a PP
        "PP": [["prep", "N"], 1],  # PPs always become prepositions followed by NPs
    })

    # Set independent probabilistic rules, e.g. pluralization, past tense
    mylang.set_unconditioned_rules({
        "sNP": [["NP"], "nom", 1],  # Subject NPs take the nominative
        "N": [["noun"], "sg", 0.8, "pl", 0.2]  # Nouns may be singular or plural
    })

    # Set the agreement rules
    mylang.set_agreement_rules({
        "verb": [["nom", "noun"], [["sg", "pl"], ["1st", "2nd", "3rd"]]]
    })

    # Generate 2 determiners
    mylang.generate_words(num_words=2, part_of_speech="det", paradigm='main')
    # Generate 10 prepositions
    mylang.generate_words(num_words=10, part_of_speech="prep", paradigm='uninflected')
    # Generate 200 adjectives
    mylang.generate_words(num_words=200, part_of_speech="adj", paradigm='uninflected')

    # Return the language
    return mylang


if __name__ == "__main__":
    # Currently with sanity check settings
    # basic creates one regular paradigm
    train_sentences_basic = main_basic()
    make_test_sentences_basic(train_sentences_basic)
    # verb_classes creates two regular paradigms, where the class of the verb is not predictable
    ## train_sentences_classes = main_verb_classes()
    ## make_test_sentences_classes(train_sentences_classes)
    # non_suppletive_allomorphy creates one regular paradigm with regular CV allomorphy
    ## train_sentences_non_suppletive_allomorphy = main_non_suppletive_allomorphy()
    ## make_test_sentences_non_suppletive_allomorphy(train_sentences_non_suppletive_allomorphy)
