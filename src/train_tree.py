
from utils import pickle_load, pickle_dump
from TreeAndFlatModels import TreeAndFlatModels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inhibitors', default=False)
    parser.add_argument('--ratio', default="all")
    args = parser.parse_args()

    inh = "_inh" if args.inhibitors else ""
    r = args.ratio

    train_filepath = "../data/tree_data/tree_and_flat_data%s_train_r%s.pkl" % (inh, r)
    test_filepath = "../data/tree_data/tree_and_flat_data%s_test_r%s.pkl" % (inh, r)

    print("Creating Tree...")
    model = TreeAndFlatModels(train_filepath = train_filepath, test_filepath = test_filepath, min_positive_examples = 10)
    model.create_ec_numbers_tree()
    model.create_tree()
    pickle_dump(model, "../models/tree_classifier%s_r%s.pkl" % (inh, r))