import language


# Main method used to generate words
def main():
    # Create a language
    language_name = "mylang"
    mylang = language.Language()

    # Set the phonology of the language
    # TODO see if the code runs with digraphs
    # TODO try using subsets of natural classes
    phonemes = {
        "C": ["p", "b", "t", "d", "k", "g", "m", "n", "s", "j", "w", "r", "l"],
        "V": ["a", "e", "i", "o", "u"]
    }
    mylang.set_phonemes(phonemes=phonemes)

    # Set the syllables of the language
    syllables = ["CVC", "CV", "VC"]
    mylang.set_syllables(syllables=syllables)

    # Set the parts of speech of the language
    parts_of_speech = ["noun", "verb"]
    mylang.set_parts_of_speech(parts_of_speech=parts_of_speech)

    # Generate 100 nouns
    for amount, noun_property in [(1, "1st"), (1, "2nd"), (30, "3rd"), (1, "2nd"), (600, "3rd"),]:
        mylang.generate_words(num_words=amount, part_of_speech="noun", paradigm=noun_property)
    mylang.generate_words(num_words=700, part_of_speech="verb", paradigm="verb1")

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

    # Generate some sentences
    num_sentences = 150000
    sentences = mylang.generate_sentences(num_sentences)
    print("\n".join(sentences) )

    # Save the language to a given path
    mylang.dump_language(filepath="Languages/mylang/mylang.json")
    language.save_sentences(sentences=sentences, filepath=f"Languages/mylang/mylang_{num_sentences}_sentences.txt")


if __name__ == "__main__":
    main()
