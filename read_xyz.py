'''
Reading and preprocessing of dataset, dividing it in training, prediction, and holdout set.
'''
import configparser
from dataclasses import dataclass
from itertools import compress
import logging
import numpy as np
#from pprint import pformat
#from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Atom:
    '''
    Class for storing information of an atom in a molecule

    Attributes:
    -----------
    element (str): atomic element
    ff_coordinates (1D array of length 3): cartesian coordinates from force field, in Angstrom
    dft_coordinates (1D array of length 3): cartesian coordinates from DFT relaxation, in Angstrom.
    '''
    element : str
    ff_coordinates: np.ndarray
    dft_coordinates: np.ndarray


def read_xyz_file(config_file : str):
    '''
    Reads the dataset [1]

    Parameters:
    -----------
    config_file (str): configuration file: contains path of file containing the dataset

    Returns:
    --------
    molecules (list of dictionaries): Each dictionary contains:
            'molecule_id' (str): id of the molecule in the dataset.
            'atomization_energy (float)': molecule atomization energy (kcal/mol)
            'atoms' (Atom): list of atoms in the molecule

    Notes:
    ------
    A molecule with k atoms is stored as k + 2 consecutive lines,
    containing number of atoms k, molecule identifier and atomization energy y,
    and k lines giving element type and coordinates (Ångström) of each atom -
    left block has coordinates from force field, right block has coordinates
    from DFT relaxation.

    References:
    ----------
    [1] Rupp, M. Int. J. Quantum Chem. 2015, 115, 1058– 1073.
    DOI: 10.1002/qua.24954 (Supporting information)
    '''

    config = configparser.ConfigParser()
    config.read(config_file)
    filename = config.get('File', 'path')

    molecules = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            num_atoms = int(lines[i].strip())
            i += 1
            molecule_id, atomization_energy = lines[i].strip().split()
            i += 1

            atoms = []
            for j in range(num_atoms):
                line = lines[i + j].strip().split()
                element = line[0]
                ff_coordinates = np.array([float(coord) for coord in line[1:4]])
                dft_coordinates = np.array([float(coord) for coord in line[4:]])
                atoms.append(Atom(element, ff_coordinates, dft_coordinates))

            molecules.append({'molecule_id': molecule_id,
                              'number_atoms': num_atoms,
                              'atomization_energy': float(atomization_energy),
                              'atoms': atoms})

            i += num_atoms

    return molecules


def count_non_hydrogen_atoms(molecules):
    '''
    Counts how many non hydrogen atoms each molecule contains

    Parameters:
    ----------
    molecules (list of dictionaries): See read_xyz_file

    Returns:
    --------
    num_non_hydrogen_atoms (1D list of ints): number of non hydrogen atoms
    '''

    num_non_hydrogen_atoms = np.empty(len(molecules))
    for idx, molecule in enumerate(molecules):
        atoms = molecule.get('atoms', [])
        non_hydrogen_atoms = [atom for atom in atoms if atom.element != 'H']  # Access 'element' attribute directly
        num_non_hydrogen_atoms[idx] = len(non_hydrogen_atoms)
    logger.debug("number of non H atoms %s", num_non_hydrogen_atoms)
    return num_non_hydrogen_atoms


def select(molecules_lot_non_hydrogen, stride : int):
    '''
    Helper function to select training and prediction set indices

    Parameters:
    -----------
    molecules_lot_non_hydrogen (list of dictionaries): has the same type as molecules
        (see read_xyz_file)
    stride (int): one molecule every `stride` ones will be included in the training set

    Returns:
    --------
    training_ids (list of str): IDs of items forming training set
    prediction_ids (list of str): IDs of items forming prediction set

    Notes:
    ------
    Only the molecules of the more represented type (molecules_lot_non_hydrogen)
    are considered here. In addition to them, the training set will comprise
    all molecules of a less represented type (see also: split_prediction_training).

    Raises:
    -------
    ValueError: if `stride` is not positive
    '''
    if stride <= 0:
        raise ValueError("Stride value must be a positive integer")

    training_ids = [mol['molecule_id'] for i, mol in enumerate(molecules_lot_non_hydrogen) if i % stride == 0]
    prediction_ids = [mol['molecule_id'] for i, mol in enumerate(molecules_lot_non_hydrogen) if i % stride != 0]

    return training_ids, prediction_ids


def split_prediction_training(molecules, training_size = 1000, filter_number = 5):
    '''
    Split into training and prediction set

    Parameters:
    -----------
    molecules (list of dictionaries): dataset. The keys are "molecule_id" and "number_atoms".
        (In our specific case it also contains "atomization_energy", "atoms"
         - their elements and coordinates - but these fields won't be used)
    training_size (int, optional): number of items to place in training set.
        Default is 1000.
    filter_number (int, optional): items having less than this num_non_hydrogen_atoms
        will be assigned the training set because they are underrepresented in the dataset;
        above this, filtering will occur. Default is 5.

    Returns:
    --------
    training_ids (list of str): molecule IDs in the training set
    prediction_ids (list of str): molecule IDs in the prediction set

    Notes:
    ------
    The dataset is not homogeneous with respect to the number of H atoms.
    Since there are too few examples of molecules with four or fewer non-H atoms
    for reliable prediction, these are all included in the training set
    (the equivalent of doing quantum calculations for them). The remaining molecules
    are drawn randomly from all molecules with five or more non-H atoms,
    stratified by size, which is known to correlate with atomization energy.
    All molecules not in the training set are assigned to the prediction set.
    '''

    if not molecules:
        print("Error: the dataset is empty")
        return [],[]

    # Step 1: Filter molecules with 4 or fewer non-H atoms
    molecules_few_non_hydrogen = [mol for mol in molecules if count_non_hydrogen_atoms([mol]) < filter_number]
    molecules_few_non_hydrogen_ids = [mol['molecule_id'] for mol in molecules_few_non_hydrogen]
    molecules_lot_non_hydrogen = [mol for mol in molecules if count_non_hydrogen_atoms([mol]) >= filter_number]
    #num_non_hydrogen_atoms = count_non_hydrogen_atoms(molecules)
    #filter_condition= num_non_hydrogen_atoms < filter_number
    #molecules_few_non_hydrogen = list(compress(molecules, filter_condition))
    #molecules_few_non_hydrogen_ids = [mol['molecule_id'] for mol in compress(molecules, filter_condition)]
    #molecules_lot_non_hydrogen = list(compress(molecules, (not condition for condition in filter_condition)))
    #need to rewrite previous lines in more compact form

    ##molecules_few_non_hydrogen = list(item for item, condition in zip(molecules, filtering) if condition)
    ##molecules_lot_non_hydrogen = list(item for item, condition in zip(molecules, filtering) if not condition)
    k = len(molecules_few_non_hydrogen)
    logger.debug("number of molecules with few non H atoms %s",k)
    if k == len(molecules): #if all molecules have few hydrogen atoms, ...
        return molecules_few_non_hydrogen_ids, [] #TODO:select a non empty prediction set appropriately!


    # Step 2: Sort the remaining molecules by the number of atoms
    sorted_molecules = sorted(molecules_lot_non_hydrogen, key=lambda mol: mol['number_atoms'])

    # Step 3: Calculate the stride value
    stride = (len(molecules) - k) // (training_size - k)

    # Step 4: Select molecules for training and prediction sets
    training_ids, prediction_ids = select(sorted_molecules,stride)
    training_ids = molecules_few_non_hydrogen_ids + training_ids
    ##print("inside function training index is",training_set)
    return training_ids, prediction_ids

    # Step 4: Select molecules for training and prediction sets

    #training_set = []
    #prediction_set = []
    #for i, mol in enumerate(sorted_molecules):
    #    if i % stride == 0:
    #        training_set.append(mol)
    #    else:
    #        prediction_set.append(mol)

    #return training_set, prediction_set
def split_training_holdout(training_set, holdout_size = 100, filter_number = 5):
    '''
    Split training and holdout set

    Parameters:
    -----------
    training_set (list of dictionaries): items in the training set
    holdout_size (int, optional): size of holdout set. Default is 100.
    filter_number (int, optional): as a result of the training set selection,
        only items having at least this num_non_hydrogen_atoms are left for the
        the hold-out set (see also: split_prediction_training). Default is 5.

    Returns:
    --------
    filtered_training_idx (list of str): IDs of items in the filtered training set
    holdout_idx (list of str): IDs of items in the holdout set

    Notes:
    ------
    The hold-out set is used to estimate performance, i.e., it acts as a proxy
    for the 6 k prediction set, and should resemble it as closely as possible
    (in a distribution sense). Therefore, it should not include molecules
    with four or fewer non-H atoms (which are all needed for the training)
    and be stratified by number of atoms.
    '''
    if not training_set:
        print("Error: training set is empty")
        return [],[]

    # Step 1: Filter molecules in the training set with 5 or more H atoms
    num_non_hydrogen_atoms = count_non_hydrogen_atoms(training_set)
    filtering= num_non_hydrogen_atoms >= filter_number
    molecules_lot_non_hydrogen = list(compress(training_set, filtering))

    # Step 2: Sort the filtered molecules by the number of atoms
    sorted_training_set = sorted(molecules_lot_non_hydrogen, key=lambda mol: mol['number_atoms'])

    # Step 3: Select holdout_size molecules from the sorted list, stratified by the number of atoms
    num_molecules = len(sorted_training_set)
    stride = max(num_molecules // holdout_size, 1)

    holdout_idx = [mol['molecule_id'] for i, mol in enumerate(sorted_training_set) if i % stride == 0]

    # Step 4: Remove the selected molecules from the training set
    training_idx = [mol['molecule_id'] for mol in training_set]
    filtered_training_idx = [mol for mol in training_idx if mol not in holdout_idx]

    return filtered_training_idx, holdout_idx
