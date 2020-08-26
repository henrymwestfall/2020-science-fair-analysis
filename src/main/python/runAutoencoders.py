import os
import pickle

from torch import nn
import torch

from fastaHandler import *
from autoencoder import *

if __name__ == "__main__":

    # create full fasta if necessary
    os.chdir("/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python/HA")
    if not os.path.exists("full.h3n2.fasta"):
        with open("full.h3n2.fasta", "w") as main_file:
            files = filter(lambda file: "h3n2" in file, os.listdir())
            for file in files:
                with open(file, "r") as f:
                    for line in f.readlines():
                        main_file.write(line)
    # run the autoencoders

    os.chdir("/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python")

    trials = 1

    parameters = [
        [10, 0.8, "full.h3n2.fasta", 50, 10],
        [5, 0.8, "full.h3n2.fasta", 50, 10],
        [15, 0.8, "full.h3n2.fasta", 50, 10]
    ]
    for i in range(trials):
        for parameter_set in parameters:
            to_pass = [str(parameter) for parameter in parameter_set]
            command = "python3 autoencoder.py " + " ".join(to_pass)
            print(f"\nRunning `{command}`\n")
            os.system(f"{command}") # run it in terminal

    # choose the best autoencoder

    os.chdir("../models")
    waiting = True

    while waiting:
        all_models = os.listdir()
        files = []
        for fname in all_models:
            if "autoencoderModel" in fname:
                files.append(fname)
        waiting = not len(files) >= 3#len(parameters)

    print("Beginning tournament...")
    handler = FASTAHandler()
    handler.load(f"/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python/HA/full.h3n2.fasta")
    training_data = [handler.loadSequence(accession) for accession in handler.accessionList()[10000:10500]]
    dts = Dataset(data=training_data)
    prep = lambda sequence: translateToNumerical(clean(sequence))

    best = None
    best_file = ""
    record = 0.0
    for model_file in files:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
            sum_accuracy = 0.0
            tests = 0
            examples_printed = 0
            for batch in dts.batches(10, prep):
                for x in batch:
                    tests += 1
                    output = torch.round(model(x))
                    correct = (output == x).float().sum()
                    accuracy = correct / output.shape[0]
                    sum_accuracy += accuracy
            accuracy = sum_accuracy / tests
            print(f"{model_file}: {accuracy * 100}")

            if examples_printed == 0:
                print(model.current_feature_vector)
                examples_printed = 1
            if accuracy >= record:
                record = accuracy
                best = model
                best_file = model_file
    print(f"The best model is in {best_file}.")
    with open(f"/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/models/{best_file}", "rb") as source:
        sourceObj = pickle.load(source)
    with open("/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/models/best.pkl", "wb") as dest:
        pickle.dump(sourceObj, dest)
