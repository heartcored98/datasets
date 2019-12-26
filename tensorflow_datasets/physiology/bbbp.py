"""BBBP(Blood-brain Barrier Penetration)  dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@article{wu2018moleculenet,
  title={MoleculeNet: a benchmark for molecular machine learning},
  author={Wu, Zhenqin and Ramsundar, Bharath and Feinberg, Evan N and Gomes, Joseph and Geniesse, Caleb and Pappu,
          Aneesh S and Leswing, Karl and Pande, Vijay},
  journal={Chemical science},
  volume={9},
  number={2},
  pages={513--530},
  year={2018},
  publisher={Royal Society of Chemistry}
}
"""

_DESCRIPTION = """\
The Blood-brain barrier penetration (BBBP) dataset comes from a recent study 52 on the modeling and prediction of
the barrier permeability. As a membrane separating circulating blood and brain extracellular fluid, the  blood-brain
barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the barrier forms a long-standing issue
in development of drugs targeting central nervous system. This dataset includes binary labels for over 2000 compounds
on their permeability properties. Scaffold splitting is also recommended for this well-defined target.

Original Download URL: http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/BBBP.csv
"""

_URL_ = "http://moleculenet.ai/datasets-1"

""" Split URLs """
_RAND_TRAIN = "https://raw.githubusercontent.com/SoojungYang/ChemDatasetSplitter/master/bbbp/random_train_bbbp.csv"
_RAND_TEST = "https://raw.githubusercontent.com/SoojungYang/ChemDatasetSplitter/master/bbbp/random_test_bbbp.csv"
_SCAFF_TRAIN = "https://raw.githubusercontent.com/SoojungYang/ChemDatasetSplitter/master/bbbp/scaffold_train_bbbp.csv"
_SCAFF_VALID = "https://raw.githubusercontent.com/SoojungYang/ChemDatasetSplitter/master/bbbp/scaffold_valid_bbbp.csv"
_SCAFF_TEST = "https://raw.githubusercontent.com/SoojungYang/ChemDatasetSplitter/master/bbbp/scaffold_test_bbbp.csv"
_SCAFF_TOTAL = "https://raw.githubusercontent.com/SoojungYang/ChemDatasetSplitter/master/bbbp/total_scaffold_bbbp.csv"

_DOWNLOAD_URL = "https://raw.githubusercontent.com/SoojungYang/ChemDatasetSplitter/master/bbbp/modified_bbbp.csv"

_ERR_MSG = """\
  graph_option['atom_feature'] must be a function which gets 
.rdkit_Chem.rdchem.Atom object and returns np.ndarray object with shape of 
  (graph_option['num_feature'],)
  """

_ATOMS = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
          'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
          'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
          'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']


class BBBPConfig(tfds.core.BuilderConfig):
  """BuilderConfig of BBBP CONFIG."""

  # @tfds.core.disallow_positional_args
  def __init__(self, fingerprint_option, graph_option, **kwargs):
    """BuilderConfig for BBBP molecule encodeers.

    Args:
      fingerprint_option: `dict` Options for generating molecular fingerprint.
      **kwargs: keyword arguments forwarded to super.
    """
    super(BBBPConfig, self).__init__(
        version=tfds.core.Version("0.1.0"),
        **kwargs
    )

    if not (graph_option['atom_feature'] is None):
      assert callable(graph_option['atom_feature']), _ERR_MSG
      assert isinstance(graph_option['num_feature'], int), _ERR_MSG
      mol = tfds.core.lazy_imports.rdkit_Chem.MolFromSmiles("C")
      atom = mol.GetAtoms()[0]
      feature = graph_option['atom_feature'](atom)
      assert isinstance(feautre, np.ndarray), _ERR_MSG
      assert len(feature.shape) == 1, _ERR_MSG
      assert feature.shape[0] == graph_option['num_feature'], _ERR_MSG

    self.fingerprint_option = fingerprint_option
    self.graph_option = graph_option


class BBBP(tfds.core.GeneratorBasedBuilder):
  """Predict Blood-brain barrier penetration in binary classification."""

  # TODO(bbbp): Set up version.s
  VERSION = tfds.core.Version('0.1.0')

  BUILDER_CONFIGS = [
      BBBPConfig(
          name="bbbp",
          fingerprint_option={
              "algorithm": "morgan",
              "kwargs": {
                  "n_bits": 2048,
                  "radius": 2,
                  "use_chirality": False,
                  "use_bond_types": True,
                  "use_features": False,
                  "bit_info": None
              }
          },
          graph_option={
              "max_padding": True,
              "max_atoms": 1000,
              "atom_feature": None,
              "num_feature": 58
          },
          description="Generates molecular fingerprints with Morgan algorithm."
      )
  ]

  def _info(self):
    # TODO(bbbp): Specifies the tfds.core.DatasetInfo object
    n_bits = self.builder_config.fingerprint_option["kwargs"]["n_bits"]
    max_atoms = self.builder_config.graph_option[
        "max_atoms"] if self.builder_config.graph_option["max_padding"] else None
    num_feature = self.builder_config.graph_option["num_feature"]

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "smiles": tfds.features.Text(),
            "fingerprint": tfds.features.Tensor(shape=(n_bits,), dtype=tf.float64),
            "graph": {
                "feature": tfds.features.Tensor(shape=(max_atoms, num_feature), dtype=tf.float64),
                "adj": tfds.features.Tensor(shape=(max_atoms, max_atoms), dtype=tf.float64)
            },
            "label": tfds.features.ClassLabel(names=["0", "1"])
        }),
        supervised_keys=("smiles", "label"),
        homepage=_URL_,
        citation=_CITATION,
    )
  
  def _split_generators(self, dl_manager):
      """Returns SplitGenerators."""
      """
      TOTAL 2039 samples. 
      11 samples that are unable to be converted to 'mols' were removed from original dataset.
      ALL: total 2039 samples.  

      1) Random split samples (seed=123, ratio=8:2)
      TRAIN: randomly split bbbp for training (1631 samples)
      TEST: randomly split bbbp for test (408 samples)

      2) Scaffold split samples
      scaffold_train: choose samples from the largest scaffold sets (1631 samples)
      scaffold_valid: choose samples from the largest scaffold sets after train set (204 samples)
      scaffold_test: choose the remaining samples as test set (204 samples)
      scaffold: total samples, but sorted by the size of the scaffold set, from the largest to the smallest (2039 samples)
      """

      return [
          tfds.core.SplitGenerator(
              name=tfds.Split.TRAIN,
              gen_kwargs={"file": dl_manager.download(_RAND_TRAIN)},
          ),
          tfds.core.SplitGenerator(
              name=tfds.Split.TEST,
              gen_kwargs={"file": dl_manager.download(_RAND_TEST)},
          ),
          tfds.core.SplitGenerator(
              name=tfds.Split('scaffold_train'),
              gen_kwargs={"file": dl_manager.download(_SCAFF_TRAIN)},
          ),
          tfds.core.SplitGenerator(
              name=tfds.Split('scaffold_validation'),
              gen_kwargs={"file": dl_manager.download(_SCAFF_VALID)},
          ),
          tfds.core.SplitGenerator(
              name=tfds.Split('scaffold_test'),
              gen_kwargs={"file": dl_manager.download(_SCAFF_TEST)},
          ),
          tfds.core.SplitGenerator(
              name=tfds.Split('scaffold'),
              gen_kwargs={"file": dl_manager.download(_SCAFF_TOTAL)},
          ),
      ]

  def _generate_examples(self, file):
    """Yields examples."""
    with tf.io.gfile.GFile(file) as f:
      reader = csv.DictReader(f)
      for row in reader:
        yield row['num'], {
            'smiles': row['smiles'],
            'fingerprint': self._generate_molecular_fingerprint(row['smiles']),
            'graph': self._generate_molecular_graph(row['smiles']),
            'label': row['p_np']
        }

  def _generate_molecular_fingerprint(self, smiles):
    if self.builder_config.fingerprint_option["algorithm"] == "morgan":
      mol = tfds.core.lazy_imports.rdkit_Chem.MolFromSmiles(smiles)
      kwargs = {}
      for key, value in self.builder_config.fingerprint_option["kwargs"].items():
        words = key.split("_")
        kw = "".join([word if i == 0 else word.capitalize()
                      for i, word in enumerate(words)])
        kwargs[kw] = value

      fp = tfds.core.lazy_imports.rdkit_Chem_AllChem.GetMorganFingerprintAsBitVect(
          mol,
          **kwargs
      )
      arr = np.zeros((1,))
      tfds.core.lazy_imports.rdkit_Datastructs.ConvertToNumpyArray(fp, arr)
      return arr

  def _generate_molecular_graph(self, smiles):
    mol = tfds.core.lazy_imports.rdkit_Chem.MolFromSmiles(smiles)
    adj_tmp = tfds.core.lazy_imports.rdkit_Chem.rdmolops.GetAdjacencyMatrix(
        mol)

    if (self.builder_config.graph_option["max_padding"]):
      if (adj_tmp.shape[0] > self.builder_config.graph_option["max_atoms"]):
        raise Exception(
            """
            Every molecule represented by smiles must be smaller than 
            graph_option.max_atoms
            """
        )
      else:
        num_atoms = self.builder_config.graph_option["max_atoms"]
    else:
      num_atoms = adj_tmp.shape[0]

    adj = np.zeros((num_atoms, num_atoms))
    feature = np.zeros(
        (num_atoms,
         self.builder_config.graph_option["num_feature"])
    )
    feature_tmp = []
    for atom in mol.GetAtoms():
      if self.builder_config.graph_option["atom_feature"] is None:
        atom_feature = np.array(
            self._one_of_k_encoding(atom.GetSymbol(), _ATOMS) +
            self._one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
            self._one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4]) +
            self._one_of_k_encoding_unk(
                atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
            [atom.GetIsAromatic()]
        )
      else:
        atom_feature = self.builder_config.graph_option["atom_feature"](atom)
      feature_tmp.append(atom_feature)
    feature[0:len(feature_tmp), :] = feature_tmp
    adj[0:len(feature_tmp), 0:len(feature_tmp)] =\
        adj_tmp + np.eye(len(feature_tmp))

    return {"feature": feature, "adj": adj}

  def _one_of_k_encoding(self, x, allowable_set):
    if x not in allowable_set:
      raise Exception(
          "input {} not in allowable set {1}".format(x, allowable_set))
    return list(map(lambda s: 1 if x == s else 0, allowable_set))

  def _one_of_k_encoding_unk(self, x, allowable_set):
    if x not in allowable_set:
      x = allowable_set[-1]
    return list(map(lambda s: 1 if x == s else 0, allowable_set))


def _snake_to_camelcase(name):
  return "".join([n if i == 0 else n.capitalize() for i, n in enumerate(name.split("_"))])
