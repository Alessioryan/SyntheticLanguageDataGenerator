import subprocess
import os

# Only running the two multitest ones for now
three_verb_classes = "3_verb_classes"
five_verb_classes = "5_verb_classes"
eight_verb_classes = "8_verb_classes"
sixteen_verb_classes = "16_verb_classes"

languages_name = "23122024-10"
must_exist_path = f'/gpfs/gibbs/project/bowern/art69/uw_synthetic_data_llm/{languages_name}/Languages'
delete_path = f"/gpfs/gibbs/project/bowern/art69/uw_synthetic_data_llm/{languages_name}/Languages/*/output/*"
delete_intermediate_models_command = f"rm -r {delete_path}"

# Make sure the path exists
if not os.path.exists(must_exist_path):
    print(f"Error: The path '{must_exist_path}' does not exist.")
    raise FileNotFoundError(f"The specified path '{must_exist_path}' must exist.")

# Get the amount of training sentences
training_amounts = [10 ** exp for exp in range(2, 7)]  # We want to go over all the values from 2 to 6 inclusive

for training_amount in training_amounts:
    for finetuning_script, testing_script in [("run_lm_finetuning.py", "modified_test.py"),
                                              ("run_lm.py", "modified_test_from_scratch.py")]:
        # We want to delete some files after training a model
        print("Deleting files with command:", delete_intermediate_models_command)
        try:
            subprocess.run(delete_intermediate_models_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Error occurred while trying to delete files:", e)

        for agreement_type_directory in [three_verb_classes, five_verb_classes,
                                         eight_verb_classes, sixteen_verb_classes]:
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
                "--train_data_file=Languages/{}/train/{}_sentences.txt".format(agreement_type_directory,
                                                                               training_amount),
                "--per_gpu_train_batch_size=1",
                "--overwrite_output_dir",
                "--overwrite_cache"
            ]
            subprocess.run(train_command, check=True)

            for generalization_type in ["seen_roots", "unseen_roots", "one_shot"]:
                print(
                    f"\n\n\n\n\n============================TESTING ON {generalization_type}============================")
                # Run the test command on the class ungrammatical ones
                test_command_class = [
                    "python3",
                    testing_script,
                    "--train_data_file=Languages/{}/{}_sentences.txt".format(agreement_type_directory, training_amount),
                    "--output_dir=Languages/{}/output".format(agreement_type_directory),
                    "--model_type=gpt2",
                    "--model_name_or_path=gpt2",
                    "--do_eval",
                    "--eval_data_file=Languages/{}/5000_{}_grammatical.txt".format(agreement_type_directory,
                                                                                   generalization_type),
                    "--eval_data_file_1=Languages/{}/5000_{}_grammatical.txt".format(agreement_type_directory,
                                                                                     generalization_type),
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
                    "--eval_data_file=Languages/{}/5000_{}_grammatical.txt".format(agreement_type_directory,
                                                                                   generalization_type),
                    "--eval_data_file_1=Languages/{}/5000_{}_grammatical.txt".format(agreement_type_directory,
                                                                                     generalization_type),
                    "--eval_data_file_2=Languages/{}/5000_{}_ungrammatical_shift.txt".format(agreement_type_directory,
                                                                                             generalization_type),
                    "--training_amount={}_{}".format(training_amount, agreement_type_directory),
                    "--generalization_type={}_shift".format(generalization_type),
                    "--per_gpu_eval_batch_size=1"
                ]
                subprocess.run(test_command_shift, check=True)
