import json
import os
import numpy.random as nprand
import random
from collections import defaultdict
from copy import deepcopy


# Method used to save sentences to a txt file
def save_sentences(sentences, filepath):
    # Open the file write only
    with open(filepath, "w") as file:
        # We want each string to be ended with a period and followed by a new line
        file.write(".\n".join(sentences))


# Helper method used for probabilistic CFGs
def choose_state(rule, is_generation):
    choices = []
    probabilities = []
    # If it's generation, use the format "A": [["B", "C", "..."], x, ["D"], y, ...]
    # If it's not generation it's unconditioned, use the format "A": ["a", "p1", x, "p2", y, ...]
    # Either way we alternate between the choices and the probabilities, the only difference is that unconditioned
    #   starts with the state of the output
    # Iterate through the values in the list of the rule
    for i in range(0 if is_generation else 1, len(rule), 2):
        # The first element is going to be a choice
        choices.append(rule[i])
        # The second element is going to be a probability
        probabilities.append(rule[i + 1])
    # Returns the chosen option and optionally, the next state
    return random.choices(choices, probabilities)[0], None if is_generation else rule[0]


# Load a Language object form a file
def load_language(directory_path):
    # Retrieve it from JSON format
    with open(os.path.join(directory_path, "mylang.json"), "r") as file:
        data = json.load(file)
    # Create an object with that data
    new_language = Language()
    new_language.phonemes = data["phonemes"]
    new_language.syllables = data["syllables"]
    new_language.syllable_lambda = data["syllable_lambda"]
    new_language.generation_rules = data["generation_rules"]
    new_language.unconditioned_rules = data["unconditioned_rules"]
    new_language.agreement_rules = data["agreement_rules"]
    new_language.words = data["words"]
    # We want to keep the set datatype, so we have to add this line
    new_language.word_set = set(data["word_set"])
    # We changed the paradigms to have strings as keys, so now we need to undo that change
    temporary_paradigms = []
    for paradigm in data["inflection_paradigms"]:
        # Add the paradigms back in the original format, with [pos, {feature tuple: affix}]
        temporary_paradigms.append([paradigm[0],
                                    {tuple(key.split(".")): value for key, value in paradigm[1].items()}])
    new_language.inflection_paradigms = temporary_paradigms
    return new_language


# Method used to inflect the agreed_lexeme_sequences according to some rules
# agreed_lexeme_sequences is a list of agreed_lexeme_sequences, paradigms is formatted as Language.inflection_paradigms
def inflect(agreed_lexeme_sequences, paradigms):
    # Make sure that a list of lists is passed in
    assert type(agreed_lexeme_sequences[0]) is list
    inflected_sentences = []
    # For each agreed_lexeme_sequence, we want to run the agreement algorithm
    for agreed_lexeme_sequence in agreed_lexeme_sequences:
        inflected_words = []
        # Do agreement over each lexeme in the sequence
        for lexeme_with_agreement in agreed_lexeme_sequence:
            # Get the lexeme and the properties of the word
            lexeme, properties = lexeme_with_agreement
            # See if it agrees with any rules (can be 0, 1, or more)
            # A for loop is used here since it's possible that more than one rule applies
            for rule in paradigms:
                # If the pos of a rule isn't in the lexeme, we continue
                if rule[0] not in properties:
                    continue
                # Otherwise, the rule applies
                # We want to see the number of applicable inflections, since it should be equal to exactly 1
                applicable_inflections = []
                for rule_properties, inflection in rule[1].items():
                    # If the property to agree with is a single string, then we check to see if it's the properties
                    if ((type(rule_properties) == str and rule_properties in properties) or
                            # Or if the type is a tuple, we check that every property in the lexeme is in the rule
                            (type(rule_properties) == tuple and
                             set(properties) & set(rule_properties) == set(rule_properties))):
                        applicable_inflections.append(inflection)
                # There should be exactly one key that works with the agreements of the lexeme
                if len(applicable_inflections) != 1:
                    raise Exception(f"Incorrect number of applicable inflections ({len(applicable_inflections)}) "
                                    f"for {lexeme_with_agreement} "
                                    f"given rule {rule}. \n"
                                    f"Debug info: properties = {properties}, "
                                    f"rule_properties = {rule_properties}, "
                                    f"applicable_indlections = {applicable_inflections}")
                # We apply the affix to the lexeme
                affix = applicable_inflections[0]
                # The affixed form depends on the position of the dash.
                # For simple affixes, we just attach it where the dash it
                lexeme = affix.replace("-", lexeme)
            # Once we go through all the rules, we can add it to inflected words
            # We add them with the properties in case we want to look at them
            inflected_words.append([lexeme, properties])
        # We now take the inflected words and turn them into the surface form
        inflected_sentence = " ".join([word_and_properties[0] for word_and_properties in inflected_words])
        # Add this to the inflected sentences
        inflected_sentences.append(inflected_sentence)
    # Return the inflected sentences
    return inflected_sentences


# Language class used to generate words according to a distribution
class Language:
    # Constructor
    def __init__(self, language=None):
        # All the fields depend on whether language is passed in
        # Maps C and V to a list of phonemes
        self.phonemes = {} if language is None else deepcopy(language.phonemes)
        self.syllables = [] if language is None else deepcopy(language.syllables)
        self.syllable_lambda = 1 if language is None else deepcopy(language.syllable_lambda)
        self.generation_rules = {} if language is None else deepcopy(language.generation_rules)
        self.unconditioned_rules = {} if language is None else deepcopy(language.unconditioned_rules)
        self.agreement_rules = {} if language is None else deepcopy(language.agreement_rules)
        self.words = {} if language is None else deepcopy(language.words)
        self.word_set = set() if language is None else deepcopy(language.word_set)
        self.inflection_paradigms = [] if language is None else deepcopy(language.inflection_paradigms)

    # Set the phonemes
    def set_phonemes(self, phonemes):
        self.phonemes = phonemes

    # Set the syllables
    def set_syllables(self, syllables):
        self.syllables = syllables

    # Set the lambda for the number of syllables in the language. The number is automatically summed to 1.
    def set_syllable_lambda(self, syllable_lambda=1):
        self.syllable_lambda = syllable_lambda

    # Set part of speech
    # Not encoded in a separate variable
    def set_parts_of_speech(self, parts_of_speech):
        for part_of_speech in parts_of_speech:
            self.words[part_of_speech] = []

    # Set sentence generation rules according to CFGs
    # The format of a rule is "A": [["B", "C", "..."], x, ["D"], y, ...]
    #   A is the input state
    #   B, C, ... are the outputs states
    #   x is the probability of the rule taking place
    #   D is another output state that takes place with probability y
    # An example is "S": [["sN", "VP"], 1]
    #   This means sentence goes to subject noun and verb phrase with probability 1
    # Another example is "VP": [["verb", "oN"], 0.7, ["verb"], 0.3]
    #   This means verb phrases take an object noun with probability 0.7
    # The probabilities must sum to 1, but this isn't checked
    def set_generation_rules(self, generation_rules):
        self.generation_rules.update(generation_rules)

    # Sets sentence generation rules for individual words that are not conditioned by other words
    # The format of a rule is "A": [["a"], "p1", x, "p2", y, ...]
    #   A is the input state
    #   a is the output state, it must be wrapped in a list
    #   x is the probability of a having property p1, y is the probability of having property p2
    # An example is "sN": [["noun"], "sing", 0.8, "pl", 0.2]
    #   This means that subject nouns map to nouns, with the feature singular with probability 0.8 and plural with 0.2
    def set_unconditioned_rules(self, unconditioned_rules):
        self.unconditioned_rules.update(unconditioned_rules)

    # Sets the agreement rules for words with a property or terminal
    # The format of a rule is "t": [["p1", "p2", ...], [["q1", "q2", ...], ["r1", ...], ...]]
    #   t is the property or terminal that takes the agreement affixes
    #   p1, p2, ... are the agreement properties that a word that t agrees with must have to trigger agreement
    #       There must be exactly one word with ALL the properties that are needed to trigger agreement
    #       Otherwise, an error is raised
    #   ["q1", "q2", ...] is a set of properties that determine feature q
    #       The word that satisfies the agreement properties must have exactly one of the features in q
    #       Otherwise, an error is raised
    #   ["r1", ...], ... are other sets of properties that determine other features
    # An example is "verb": [["nom"], [["sg", "pl"], ["1st", "2nd", "3rd"]]]
    #   This means that verbs must agree with a word that has the property nominative
    #   Then, the verb agrees with that word, taking either the property singular or plural, and 1st, 2nd, or 3rd
    # The agreement rules don't dictate the inflections, just what words agree with what
    # FOR NOW, WORDS CAN ONLY AGREE WITH ONE OTHER WORD. POTENTIALLY CHANGE THIS LATER.
    def set_agreement_rules(self, agreement_rules):
        self.agreement_rules.update(agreement_rules)

    # Define an inflection pattern for a given paradigm
    # The format of a rule is ["w", {("p1", "p2", ...): "-s1", ("q1", ...): "-s2", ...}]
    #   w is the property of the word that triggers a check for whether this word inflects for this rule
    #   ("p1", "p2", ...) is a tuple of properties that result in inflection being triggered
    #       There must be exactly one tuple that triggers inflection for a given word
    #       Otherwise, an error is raised
    #   -s1 is a suffix that is appended to the word if the properties that trigger inflection are met
    #       Prefixes (pref-) and circumfixes (cir-cum) are also supported
    #   ("q1", ...): "-s2" is another tuple of properties that triggers inflection with suffix "-s2"
    # If we call ["w", {("p1", "p2", ...): "-s1", ("q1", ...): "-s2", ...}] R1, the format of the input is [R1, R2, ...]
    #   The rules apply in order
    #   If you're adding a single inflection paradigm, wrap the rule in a list
    # An example for one rule is ["noun", {"sg": "-", "pl": "-ol"}]
    #   For this rule, we see that nouns must agree with either singular of plural. If they agree with none or both,
    #       an error is thrown
    #   Whichever feature it agrees with results in that suffix getting added. The singular is unmarked while the
    #       plural takes the suffix "-ol"
    def set_inflection_paradigms(self, inflection_paradigms):
        # The inflection_paradigm is a dictionary. All inflections are suffixes
        self.inflection_paradigms.extend(inflection_paradigms)

    # Add words to our lexicon at the end of the list for that part of speech
    # Words can be individual words or tuples consisting of (word, paradigm_number)
    # Paradigm is a mandatory string parameter which defines the words as being part of that paradigm
    # Returns a list containing the new words
    # If you want to include more than one property in the paradigm, separate them with periods
    def generate_words(self, num_words, part_of_speech, paradigm):
        # Generate words with each phoneme in a given class appearing with the same frequency
        # All syllable types appear with equal frequency too
        # We use a while loop since we don't want duplicate words (for now!)
        new_words = []
        while len(new_words) < num_words:
            # Select a random number of syllables (+1 for non-empty syllables)
            num_syllables = nprand.poisson(self.syllable_lambda) + 1
            # For every syllable, choose a random syllable structure and construct a word out of it
            word = ""
            for syllable in range(num_syllables):
                syllable_structure = random.choice(self.syllables)
                # For every natural class in the syllable, choose a random phoneme that fits that description
                for natural_class in syllable_structure:
                    # Find a random phoneme from that natural class and add it to the word
                    word += (nprand.choice(self.phonemes[natural_class]))
            # If we generated a new word, we add it to our lexicon and to the words we made
            if word not in self.word_set:
                new_words.append((word, paradigm))
                self.word_set.add(word)
        # Add the new_words to the part of speech they were made for
        self.words[part_of_speech] += new_words
        # Now return the list in case it's needed
        return new_words

    # Generate sentences with a Zipfian distribution
    # Required words is by default None.
    #   If you want to generate sentences with words from a specific set, you pass in a dictionary.
    #   This dictionary maps pos of words to a list tuples of words and paradigms
    #   For example, required_words = {pos: []}
    # If you're generating sentences with required_words, note that all parts of speech not in required_words will
    #   be drawn with Zipf's distribution as normal. This may mean that if a sentence is generated with no terminal
    #   pos in required words, then there won't be any words from required words in the sentence, and that if a
    #   sentence has more than one terminal pos in required words, all of those will be drawn from required words.
    def generate_sentences(self, num_sentences, required_words=None):
        # Prepare the sentences we want
        sentences = []
        agreed_lexeme_sequences = []
        for _ in range(num_sentences):
            # GENERATE THE TERMINAL POS STATES AND PROPERTIES
            # We want the sentence to only contain terminal nodes, but we start with the start state
            sentence = "S-"
            # All terminals and properties are all lowercase.
            # If the sentence contains uppercase letters, then we have not finished constructing it
            while sentence.lower() != sentence:
                # Continually replace non-terminal nodes until the sentence doesn't change
                temp_sentence = ""
                for state in sentence.strip().split(" "):
                    # Split this state from any properties it has
                    raw_state, existing_properties = state.split("-")
                    # If the raw states is a terminal part of speech, continue the loop
                    if raw_state in self.words.keys():
                        # We want to add the whole state
                        temp_sentence += state + " "
                        continue
                    # If it's a generation rule, then we add words individually to the temp_sentence
                    if raw_state in self.generation_rules:
                        # Choose the next state(s) for this state
                        next_states, new_property = choose_state(rule=self.generation_rules[raw_state],
                                                                 is_generation=True)
                    # If it's an unconditioned rule, we see if we want to add any properties
                    elif raw_state in self.unconditioned_rules:
                        # Choose the property for this state
                        new_property, next_states = choose_state(rule=self.unconditioned_rules[raw_state],
                                                                 is_generation=False)
                    # Sanity check: if we enter this loop, the state should be in either generation or unconditioned
                    else:
                        raise Exception(f"Invalid state for input {state}. Check code.")
                    # For each next state, add the new property
                    for next_state in next_states:
                        temp_sentence += (f'{next_state}-'
                                          f'{existing_properties + "." if existing_properties else ""}'
                                          f'{new_property if new_property else ""} ')
                # Update the value of sentence
                sentence = temp_sentence

            # REPLACE THE TERMINAL POSs WITH WORDS GENERATED ACCORDING TO THE ZIPFIAN DISTRIBUTIAN
            # Start with splitting the word into its pieces again
            preagreement_words = []
            for preagreement_word in sentence.strip().split(" "):
                # Split each word into the lexeme and its properties
                terminal, properties = preagreement_word.split("-")
                # Add the lexeme and a list of properties to the preagreement words
                preagreement_words.append([terminal, properties.split(".")])
            preagreement_lexemes = []
            for preagreement_word in preagreement_words:
                # Get the terminal part of speech (pos) and the properties of the word
                pos, properties = preagreement_word
                # Generate a word according to Zipf's distribution
                skew = 1.2  # Note: This parameter can be changed. Find a naturalistic one
                # There isn't a nice way to generate an index according to Zipf's law
                # The way we do it here is we generate a random index according to an unbounded distribution
                # If it is outside the range of our list, we generate another one
                # Otherwise, we use it
                index = -1
                # If there are no word which we are required to use, then we're good!
                # If there are required words but the part of speech is not in required words, we get a word as normal
                if required_words is None or pos not in required_words:
                    while index == -1:
                        # We generate an index, subtracting 1 since Zipf's starts from 1
                        index = nprand.zipf(skew, 1)[0] - 1
                        # If it's out of the range, we reset the index to 0
                        if index >= len(self.words[pos]):
                            index = -1
                    # If it is in the range, we exit our loop and get the word and paradigm
                    word, paradigm = self.words[pos][index]
                # If we want to generate words from a list of words, then we draw uniformly from that set
                else:
                    # Get the words at random from the list
                    word, paradigm = random.choice(required_words[pos])
                # Add the sentence to the word_sentence
                # We also make the part of speech and the existing paradigm a new feature
                # We use paradigm.split(".") since if an entry has more than one property we mark them with . boundaries
                preagreement_lexemes.append([word, properties + [pos] + paradigm.split(".")])

            # ADD AGREEMENT PROPERTIES, NOT YET INFLECTING
            # Now we iterate over every word to see if it must agree with any other words
            agreed_words = []
            for preagreement_word in preagreement_lexemes:
                # Check to see if there's a rule describing this word.
                # If there isn't, out work is done, so we add it to agreed_words and continue
                # If none of the properties of a word are in the agreement rules
                agreement_properties = list(set(preagreement_word[1]) & set(self.agreement_rules))
                if len(agreement_properties) == 0:
                    agreed_words.append(preagreement_word)
                    continue
                # For now, we can only handle one agreement. We might change this later
                assert len(agreement_properties) == 1
                # Get the rule for that terminal
                rule = self.agreement_rules[agreement_properties[0]]
                required_properties = set(rule[0])
                # If there is, we want to find THE word that triggers agreement
                words_triggering_agreement = []
                for other_word in preagreement_lexemes:
                    # Make sure it has all the right properties
                    if required_properties & set(other_word[1]) == required_properties:
                        # If it does have all the right properties, add it to the list of words triggering agreement
                        words_triggering_agreement.append(other_word)
                # Now we make sure there's EXACTLY ONE word triggering agreement
                if len(words_triggering_agreement) != 1:
                    raise Exception(f"{len(words_triggering_agreement)} words triggered agreement for "
                                    f"{preagreement_word}. These words are {words_triggering_agreement}. "
                                    f"The rule that triggered it is {rule}. "
                                    f"Check rules.")
                # Perfect! Now that we found the word that agrees with our preagreement word, we find the properties
                # We want to check that the word triggering agreement has one of each property in each set
                word_triggering_agreement = words_triggering_agreement[0]
                # For every property that our preagreement word seeks, we look to see if trigger word has it
                new_properties = []
                # For every feature in that the preagreement word looks for
                for sought_feature in rule[1]:
                    # Make sure that the intersection of the sought properties with the word triggering agreement is one
                    property_intersection = list(set(sought_feature) & set(word_triggering_agreement[1]))
                    # If there isn't exactly 1, then raise an error
                    if len(property_intersection) != 1:
                        raise Exception(f"Incorrect number of properties found for {preagreement_word}. "
                                        f"Sought feature: {sought_feature}. "
                                        f"Word triggering agreement: {word_triggering_agreement}")
                    # If there is exactly one feature, then we add it to the new properties of the preagreement word
                    new_properties.append(property_intersection[0])
                # Now we have found all the new properties of the preagreement word!
                # All that is left is to join the old and new properties, and add this word to our agreed word list
                agreed_words.append([preagreement_word[0], preagreement_word[1] + new_properties])
            agreed_lexeme_sequences.append(agreed_words)

            # MAKE EACH WORD HAVE INFLECTIONS
            # We get only the first element of the output since we are only making one sentence
            inflected_words = inflect([agreed_words], self.inflection_paradigms)[0]

            # FINALLY, GIVE THE SURFACE FORM
            # We only add the final sentence, not the properties, but we keep them along until the end for debugging
            sentences.append(inflected_words)
        # Finally, we exit the loop and return the list of sentences
        return sentences, agreed_lexeme_sequences

    # Save the language in a given file
    def dump_language(self, directory_path):
        # Make the path to the file, if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        # Store it in JSON format
        with open(os.path.join(directory_path, "mylang.json"), 'w') as file:
            # Pretty much all data can be stored as is
            data = {
                "phonemes": self.phonemes,
                "syllables": self.syllables,
                "syllable_lambda": self.syllable_lambda,
                "generation_rules": self.generation_rules,
                "unconditioned_rules": self.unconditioned_rules,
                "agreement_rules": self.agreement_rules,
                "words": self.words,
                # Word_set must first be converted to a list
                "word_set": list(sorted(self.word_set) )
            }
            # Inflection paradigms has tuples as keys.
            # We want to replace each tuple with a string with dots separating the properties
            temporary_paradigms = []
            for paradigm in self.inflection_paradigms:
                # The first value of every rule should be the same
                # The second value is the same as well, except each key is now a string with periods between properties
                # We only join the different parts with . if the key is a tuple
                temporary_paradigms.append([paradigm[0], {(".".join(key) if type(key) == tuple else key): value
                                                          for key, value in paradigm[1].items()}])
            # Don't forget to add it to the datafile!
            data["inflection_paradigms"] = temporary_paradigms
            # Dump the datafile
            json.dump(data, file)