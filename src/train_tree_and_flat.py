import argparse
import os

from utils import pickle_load, pickle_dump
from TreeAndFlatModels import TreeAndFlatModels

MIN_POSITIVE_EXAMPLE = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inhibitors', default="False")
    parser.add_argument('--similarity', default="True")
    parser.add_argument('--ratio', default="all")
    parser.add_argument('--output_folder', default='./')
    args = parser.parse_args()

    if args.similarity.lower() == "true":
        sim = "_sim"
    elif args.similarity.lower() == "false":
        sim = ""
    else:
        print("Argument Error: --similarity must be given a valid boolean identifier.")
        exit(1)

    if args.inhibitors.lower() == "true":
        inh = "_inh"
    elif args.inhibitors.lower() == "false":
        inh = ""
    else:
        print("Argument Error: --inhibitors must be given a valid boolean identifier.")
        exit(1)

    r = args.ratio
    output_folder = args.output_folder

    train_filepath = "../data/tree_data/tree_and_flat_data%s_train_r%s.pkl" % (inh, r)
    test_filepath = "../data/tree_data/tree_and_flat_data%s_test.pkl" % (inh)

    print("Creating Tree...")
    model = TreeAndFlatModels(train_filepath = train_filepath, test_filepath = test_filepath, similarity= True, min_positive_examples = MIN_POSITIVE_EXAMPLE)
    model.create_ec_numbers_tree()
    model.create_tree()
    print("Dumping Tree...")
    pickle_dump(model, os.path.join(output_folder, "../models/tree_classifier%s%s_r%s.pkl" % (sim, inh, r)))
