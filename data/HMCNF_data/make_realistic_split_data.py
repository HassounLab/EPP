
import pickle
import numpy as np
import argparse

def pickle_load(filepath):
    f = open(filepath, "rb")
    data = pickle.load(f)
    f.close()
    return data

def pickle_dump(data, filepath):
    f = open(filepath, "wb")
    pickle.dump(data, f)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inhibitors', default="False")
    parser.add_argument('--similarity', default="True")
    args = parser.parse_args()

    if args.similarity.lower() == "true":
        sim = "_morgan2binary_sim" # "_maccs_sim"
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


    data = pickle_load("data_morgan2binary.pkl")
    Pl1  = pickle_load("Pl1.pkl")
    Pl2  = pickle_load("Pl2.pkl")
    Pl3  = pickle_load("Pl3.pkl")
    Pl4  = pickle_load("Pl4.pkl")
    Pg   = pickle_load("Pg.pkl")
    Pl1_weights = pickle_load("Pl1%s_bal_weights%s.pkl" % (sim, inh))
    Pl2_weights = pickle_load("Pl2%s_bal_weights%s.pkl" % (sim, inh))
    Pl3_weights = pickle_load("Pl3%s_bal_weights%s.pkl" % (sim, inh))
    Pl4_weights = pickle_load("Pl4%s_bal_weights%s.pkl" % (sim, inh))
    Pg_weights  = pickle_load("Pg%s_bal_weights%s.pkl" % (sim, inh))

    print('Original shapes:')
    print(Pl1.shape)
    print(Pl2.shape)
    print(Pl3.shape)
    print(Pl4.shape)
    print(Pg.shape)
    print()

    # remove pre-computed ECs that have too few example, removing
    ec_i_to_exclude = pickle_load("ec_i_to_exclude_realistic.pkl")
    subsubclass_i_to_exclude = pickle_load("subsubclass_i_to_exclude_realistic.pkl")
    subclass_i_to_exclude = pickle_load("subclass_i_to_exclude_realistic.pkl")

    Pl4 = np.delete(Pl4, ec_i_to_exclude, 1)
    Pl4_weights = np.delete(Pl4_weights, ec_i_to_exclude, 1)

    Pl3 = np.delete(Pl3, subsubclass_i_to_exclude, 1)
    Pl3_weights = np.delete(Pl3_weights, subsubclass_i_to_exclude, 1)

    Pl2 = np.delete(Pl2, subclass_i_to_exclude, 1)
    Pl2_weights = np.delete(Pl2_weights, subclass_i_to_exclude, 1)

    Pg = np.hstack((Pl4, Pl3, Pl2, Pl1)) 
    Pg_weights = np.hstack((Pl4_weights, Pl3_weights, Pl2_weights, Pl1_weights))


    print('Saving realistic split (a.k.a. reduced dataset) labels and weights...')
    pickle_dump(Pl1, "Pl1__realistic.pkl")
    pickle_dump(Pl1_weights, "Pl1%s_bal_weights%s__realistic.pkl" % (sim, inh))
    pickle_dump(Pl2, "Pl2__realistic.pkl")
    pickle_dump(Pl2_weights, "Pl2%s_bal_weights%s__realistic.pkl" % (sim, inh))
    pickle_dump(Pl3, "Pl3__realistic.pkl")
    pickle_dump(Pl3_weights, "Pl3%s_bal_weights%s__realistic.pkl" % (sim, inh))
    pickle_dump(Pl4, "Pl4__realistic.pkl")
    pickle_dump(Pl4_weights, "Pl4%s_bal_weights%s__realistic.pkl" % (sim, inh))
    pickle_dump(Pg, "Pg__realistic.pkl")
    pickle_dump(Pg_weights, "Pg%s_bal_weights%s__realistic.pkl" % (sim, inh))

    print('Shapes:')
    print(Pl1.shape)
    print(Pl2.shape)
    print(Pl3.shape)
    print(Pl4.shape)
    print(Pg.shape)
    print()


    ec_i_to_exclude_val = pickle_load("ec_i_to_exclude_realistic_val.pkl")
    subsubclass_i_to_exclude_val = pickle_load("subsubclass_i_to_exclude_realistic_val.pkl")
    subclass_i_to_exclude_val = pickle_load("subclass_i_to_exclude_realistic_val.pkl")

    Pl4 = np.delete(Pl4, ec_i_to_exclude_val, 1)
    Pl4_weights = np.delete(Pl4_weights, ec_i_to_exclude_val, 1)

    Pl3 = np.delete(Pl3, subsubclass_i_to_exclude_val, 1)
    Pl3_weights = np.delete(Pl3_weights, subsubclass_i_to_exclude_val, 1)

    Pl2 = np.delete(Pl2, subclass_i_to_exclude_val, 1)
    Pl2_weights = np.delete(Pl2_weights, subclass_i_to_exclude_val, 1)   
    
    Pg = np.hstack((Pl4, Pl3, Pl2, Pl1))
    Pg_weights = np.hstack((Pl4_weights, Pl3_weights, Pl2_weights, Pl1_weights))

    print('Saving realistic split (a.k.a. reduced dataset) FOR VALIDATION PURPOSES labels and weights...')
    pickle_dump(Pl1, "Pl1__realistic_for_validation.pkl")
    pickle_dump(Pl1_weights, "Pl1%s_bal_weights%s__realistic_for_validation.pkl" % (sim, inh))
    pickle_dump(Pl2, "Pl2__realistic_for_validation.pkl")
    pickle_dump(Pl2_weights, "Pl2%s_bal_weights%s__realistic_for_validation.pkl" % (sim, inh))
    pickle_dump(Pl3, "Pl3__realistic_for_validation.pkl")
    pickle_dump(Pl3_weights, "Pl3%s_bal_weights%s__realistic_for_validation.pkl" % (sim, inh))
    pickle_dump(Pl4, "Pl4__realistic_for_validation.pkl")
    pickle_dump(Pl4_weights, "Pl4%s_bal_weights%s__realistic_for_validation.pkl" % (sim, inh))
    pickle_dump(Pg, "Pg__realistic_for_validation.pkl")
    pickle_dump(Pg_weights, "Pg%s_bal_weights%s__realistic_for_validation.pkl" % (sim, inh))

    print('Shapes:')
    print(Pl1.shape)
    print(Pl2.shape)
    print(Pl3.shape)
    print(Pl4.shape)
    print(Pg.shape)
    print()

