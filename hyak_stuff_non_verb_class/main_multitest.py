import subprocess

# Only running the two multitest ones for now
one_regular_paradigm_dir = "one_regular_paradigm"
two_regular_classes_dir = "two_verb_classes"
non_suppletive_allomorphy_dir = "non_suppletive_allomorphy"

# Get the amount of training sentences
training_amounts = [10 ** exp for exp in range(2, 6)]  # We want to go over all the values from 2 to 6 inclusive

for agreement_type_directory in [one_regular_paradigm_dir, two_regular_classes_dir, non_suppletive_allomorphy_dir]:
    for training_amount in training_amounts:
        print(f"\n\n\n\n\n============================TRAINING ON {training_amount}============================")
        # Install transformers
        subprocess.run(["pip", "install", "transformers"], check=True)
        # Train the model
        train_command = [
            "python3",
            "run_lm_finetuning.py",
            "--output_dir=Languages/{}/output".format(agreement_type_directory),
            "--model_type=gpt2",
            "--model_name_or_path=gpt2",
            "--save_total_limit=3",
            "--do_train",
            "--train_data_file=Languages/{}/train/{}_sentences.txt".format(agreement_type_directory, training_amount),
            "--per_gpu_train_batch_size=1",
            "--overwrite_output_dir",
            "--overwrite_cache"
        ]
        subprocess.run(train_command, check=True)

        for generalization_type in ["seen_roots", "unseen_roots", "one_shot"]:
            print(f"\n\n\n\n\n============================TESTING ON {generalization_type}============================")
            # Run the test command on the class ungrammatical ones
            # Run this if we're not in the one regular paradigm
            if agreement_type_directory != one_regular_paradigm_dir:
                test_command_class = [
                    "python3",
                    "modified_test.py",
                    "--train_data_file=Languages/{}/{}_sentences.txt".format(agreement_type_directory, training_amount),
                    "--output_dir=Languages/{}/output".format(agreement_type_directory),
                    "--model_type=gpt2",
                    "--model_name_or_path=gpt2",
                    "--do_eval",
                    "--eval_data_file=Languages/{}/5000_{}_grammatical.txt".format(agreement_type_directory, generalization_type),
                    "--eval_data_file_1=Languages/{}/5000_{}_grammatical.txt".format(agreement_type_directory, generalization_type),
                    "--eval_data_file_2=Languages/{}/5000_{}_ungrammatical_class.txt".format(agreement_type_directory,
                                                                                   generalization_type),
                    "--training_amount={}_{}".format(training_amount, agreement_type_directory),
                    "--generalization_type={}_class".format(generalization_type),
                    "--per_gpu_eval_batch_size=1"
                ]
                subprocess.run(test_command_class, check=True)

            # Run the test command on the shift ungrammatical ones
            test_command_shift = [
                "python3",
                "modified_test.py",
                "--train_data_file=Languages/{}/{}_sentences.txt".format(agreement_type_directory, training_amount),
                "--output_dir=Languages/{}/output".format(agreement_type_directory),
                "--model_type=gpt2",
                "--model_name_or_path=gpt2",
                "--do_eval",
                "--eval_data_file=Languages/{}/5000_{}_grammatical.txt".format(agreement_type_directory, generalization_type),
                "--eval_data_file_1=Languages/{}/5000_{}_grammatical.txt".format(agreement_type_directory, generalization_type),
                "--eval_data_file_2=Languages/{}/5000_{}_ungrammatical_shift.txt".format(agreement_type_directory,
                                                                               generalization_type),
                "--training_amount={}_{}".format(training_amount, agreement_type_directory),
                "--generalization_type={}_shift".format(generalization_type),
                "--per_gpu_eval_batch_size=1"
            ]
            subprocess.run(test_command_shift, check=True)
