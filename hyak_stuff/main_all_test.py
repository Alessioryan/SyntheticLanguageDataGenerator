import subprocess
import os

# Only running the two multitest ones for now
one_regular_paradigm_dir = "one_regular_paradigm"
two_regular_classes_dir = "two_verb_classes"
non_suppletive_allomorphy_dir = "non_suppletive_allomorphy"

languages_name = "2024Oct23runs10"
must_exist_path = f'/mmfs1/gscratch/zlab/tosolini/{languages_name}/hyak_stuff/Languages'
delete_path = f"/mmfs1/gscratch/zlab/tosolini/{languages_name}/hyak_stuff/Languages/*/output/*"
delete_intermediate_models_command = [
    "rm",
    "-r",
    delete_path
]

# Make sure the path exists
if not os.path.exists(must_exist_path):
    print(f"Error: The path '{must_exist_path}' does not exist.")
    raise FileNotFoundError(f"The specified path '{must_exist_path}' must exist.")

# Get the amount of training sentences
training_amounts = [10 ** exp for exp in range(2, 7)]  # We want to go over all the values from 2 to 6 inclusive

for finetuning_script, testing_script in [("run_lm_finetuning.py", "modified_test.py"),
                                          ("run_lm.py", "modified_test_from_scratch.py")]:
    for training_amount in training_amounts:
        # We want to delete some files after training a model
        print("Deleting files with command:", " ".join(delete_intermediate_models_command))
        try:
            subprocess.run(delete_intermediate_models_command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error occurred while trying to delete files:", e)

        for agreement_type_directory in [one_regular_paradigm_dir, two_regular_classes_dir, non_suppletive_allomorphy_dir]:
            print(f"\n\n\n\n\n============================TRAINING ON {training_amount}============================")
            # Install transformers
            subprocess.run(["pip", "install", "transformers"], check=True)
            # Train the model
            train_command = [
                "python3",
                finetuning_script,
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
                        testing_script,
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
                    testing_script,
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
