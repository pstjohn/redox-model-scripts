import numpy as np
import pandas as pd
import tensorflow as tf
import nfp

from tqdm import tqdm
                
from preprocessor import preprocessor

redf_spin = pd.read_csv('/projects/rlmolecule/pstjohn/atom_spins/redf_spins_expanded.csv.gz')
redf_bv = pd.read_csv('/projects/rlmolecule/svss/Project-Redox/bur-vol_data_water/buried_volumes_water_all.csv.gz', index_col=0).drop(['Unnamed: 0.1'], 1)

redf_spin = redf_spin[redf_spin.atom_type != 'H']
redf = redf_spin.merge(redf_bv, on=['smiles', 'atom_index'], how='inner')

new_data = pd.read_csv('/projects/rlmolecule/pstjohn/spin_gnn/20210211_fixed_rl_spin_bv_data.csv.gz')
new_data = new_data.rename(columns={'fractional_spin': 'spin'})[['smiles', 'atom_index', 'spin', 'buried_vol']]

if __name__ == '__main__':

    # Get a shuffled list of unique SMILES
    redf_smiles = redf.smiles.unique()
    new_smiles = new_data.smiles.unique()
    
    rng = np.random.default_rng(1)
    rng.shuffle(redf_smiles)
    rng.shuffle(new_smiles)

    # split off 5000 each for test and valid sets
    test, valid, train = np.split(redf_smiles, [5000, 10000])
    test_new, train_new = np.split(new_smiles, [500])

    # Save these splits for later
    np.savez_compressed('split_spin_bv.npz', train=train, valid=valid, test=test, test_new=test_new, train_new=train_new)

    redf_train = redf[redf.smiles.isin(train)]
    redf_valid = redf[redf.smiles.isin(valid)]
    
    new_train = new_data[new_data.smiles.isin(train_new)]
    new_test = new_data[new_data.smiles.isin(test_new)]    

    def inputs_generator(df, train=True):

        for smiles, idf in tqdm(df.groupby('smiles')):
            input_dict = preprocessor.construct_feature_matrices(smiles, train=train)
            spin = idf.set_index('atom_index').sort_index().spin
            fractional_spin = spin.abs() / spin.abs().sum()
            input_dict['spin'] = fractional_spin.values
            input_dict['bur_vol'] = idf.buried_vol.values            
            
            assert len(fractional_spin.values) == input_dict['n_atom']

            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()

    
    serialized_train_new_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(new_train, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords_spin_bv/train_new.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_new_dataset)

    serialized_valid_new_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(new_test, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords_spin_bv/valid_new.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_new_dataset)

               
            
    serialized_train_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(redf_train, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords_spin_bv/train.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)

    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(redf_valid, train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords_spin_bv/valid.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)
    

