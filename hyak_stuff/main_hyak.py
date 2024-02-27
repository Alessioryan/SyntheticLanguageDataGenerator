# Set all the directories at once
one_regular_paradigm_directory = "one_regular_paradigm"

# Only running the regular paradigm for now
# two_regular_classes_directory = "Nov-30-two-regular-classes"
# non_suppletive_allomorphy_directory = "Dec-1-non-suppletive-allomorphy"


import subprocess

# Get the amount of training sentences
training_amounts = [1000000]

for training_amount in training_amounts:
    print(f"\n\n\n\n\n============================TRAINING ON {training_amount}============================")
    # Install transformers
    subprocess.run(["pip", "install", "transformers"], check=True)
    # Train the model
    train_command = [
        "python3",
        "run_lm_finetuning.py",
        "--output_dir={}/output".format(one_regular_paradigm_directory),
        "--model_type=gpt2",
        "--model_name_or_path=gpt2",
        "--save_total_limit=2",
        "--do_train",
        "--train_data_file={}/train/{}_sentences.txt".format(one_regular_paradigm_directory, training_amount),
        "--per_gpu_train_batch_size=1",
        "--overwrite_output_dir",
        "--overwrite_cache"
    ]
    subprocess.run(train_command, check=True)

    for generalization_type in ["seen_roots", "unseen_roots", "one_shot"]:
        print(f"\n\n\n\n\n============================TESTING ON {generalization_type}============================")
        # Run the test command
        test_command = [
            "python3",
            "modified_test.py",
            "--train_data_file={}/{}_sentences.txt".format(one_regular_paradigm_directory, training_amount),
            "--output_dir={}/output".format(one_regular_paradigm_directory),
            "--model_type=gpt2",
            "--model_name_or_path=gpt2",
            "--do_eval",
            "--eval_data_file={}/5000_{}_grammatical.txt".format(one_regular_paradigm_directory, generalization_type),
            "--eval_data_file_1={}/5000_{}_grammatical.txt".format(one_regular_paradigm_directory, generalization_type),
            "--eval_data_file_2={}/5000_{}_ungrammatical.txt".format(one_regular_paradigm_directory, generalization_type),
            "--training_amount={}".format(training_amount),
            "--generalization_type={}".format(generalization_type),
            "--per_gpu_eval_batch_size=1"
        ]
        subprocess.run(test_command, check=True)