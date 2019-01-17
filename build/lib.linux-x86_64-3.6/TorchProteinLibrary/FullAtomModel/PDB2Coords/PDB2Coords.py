# TO DO:
# 1. pdb2sequence
# 2. loading and allocating array
# 3. dealing with missing aa

import torch
from Bio.PDB import *
from Bio.PDB.Polypeptide import three_to_one
import numpy as np

import _FullAtomModel

def get_sequence(structure):
	sequence = ''
	residues = list(structure.get_residues())
	for n, residue in enumerate(residues):
		try:
			sequence += three_to_one(residue.get_resname())
		except:
			raise Exception("Can't extract sequence")
	return sequence

def pdb2sequence(filenames):
	p = PDBParser(PERMISSIVE=1, QUIET=True)
	sequences = []
	for filename in filenames:
		structure = p.get_structure('X', filename)
		sequences.append(get_sequence(structure))
	return sequences

def convertStringList(stringList):
	'''Converts list of strings to 0-terminated byte tensor'''
	maxlen = 0
	for string in stringList:
		string += '\0'
		if len(string)>maxlen:
			maxlen = len(string)
	ar = np.zeros( (len(stringList), maxlen), dtype=np.uint8)
	
	for i,string in enumerate(stringList):
		npstring = np.fromstring(string, dtype=np.uint8)
		ar[i,:npstring.shape[0]] = npstring
	
	return torch.from_numpy(ar)

def convertString(string):
	'''Converts a string to 0-terminated byte tensor'''  
	return torch.from_numpy(np.fromstring(string+'\0', dtype=np.uint8))

class PDB2CoordsOrdered:
					
	def __call__(self, filenames):
		
		self.filenamesTensor = convertStringList(filenames)
		self.num_atoms = []
		self.sequences = pdb2sequence(filenames)
		self.seqTensor = convertStringList(self.sequences)
		for seq in self.sequences:
			self.num_atoms.append(FullAtomModel.getSeqNumAtoms(seq))
		
		max_num_atoms = max(self.num_atoms)
		batch_size = len(self.num_atoms)

		output_coords_cpu = torch.zeros(batch_size, max_num_atoms*3, dtype=torch.double)
		output_resnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)
		output_atomnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)

		_FullAtomModel.PDB2CoordsOrdered(self.filenamesTensor, output_coords_cpu, output_resnames_cpu, output_atomnames_cpu)
	
		return output_coords_cpu, output_resnames_cpu, output_atomnames_cpu, self.num_atoms

class PDB2CoordsUnordered:
					
	def __call__(self, filenames):
		
		filenamesTensor = convertStringList(filenames)
		batch_size = len(filenames)
		num_atoms = torch.zeros(batch_size, dtype=torch.int)
		output_coords_cpu = torch.zeros(batch_size, 1, dtype=torch.double)
		output_resnames_cpu = torch.zeros(batch_size, 1, 1, dtype=torch.uint8)
		output_atomnames_cpu = torch.zeros(batch_size, 1, 1, dtype=torch.uint8)

		_FullAtomModel.PDB2CoordsUnordered(filenamesTensor, output_coords_cpu, output_resnames_cpu, output_atomnames_cpu, num_atoms)
	
		return output_coords_cpu, output_resnames_cpu, output_atomnames_cpu, num_atoms

class PDB2CoordsBiopython:
	def __init__(self):
		self.parser = PDBParser(PERMISSIVE=1, QUIET=True)
	
	def __call__(self, filenames, chains):
		batch_size = len(filenames)
		
		chain_structures = []
		for filename, chain in zip(filenames, chains):
			structure = self.parser.get_structure('X', filename)
			chain_structures.append(structure[0][chain])
		
		num_atoms = [0 for i in filenames]
		for n, chain in enumerate(chain_structures):
			for atom_idx, atom in enumerate(chain.get_atoms()):
				if atom.element == 'H':
					continue
				num_atoms[n] += 1
		
		max_num_atoms = max(num_atoms)
		print(num_atoms)
		output_coords_cpu = torch.zeros(batch_size, max_num_atoms*3, dtype=torch.double)
		output_resnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)
		output_atomnames_cpu = torch.zeros(batch_size, max_num_atoms, 4, dtype=torch.uint8)

		atom_idx = 0
		for n, chain in enumerate(chain_structures):
			for atom in chain.get_atoms():
				if atom.element == 'H':
					continue
				vec = atom.get_coord()
				# print(torch.from_numpy(vec).to(dtype=torch.double))
				output_coords_cpu.data[n, 3*atom_idx : 3*atom_idx+3] = torch.from_numpy(vec).to(dtype=torch.double)
				residue = atom.get_parent()
				atom_name = convertString(atom.get_name())
				print(atom_idx, atom_name, max_num_atoms)
				res_name = convertString(residue.get_resname())
				output_atomnames_cpu.data[n, atom_idx, :atom_name.size(0)].copy_(atom_name.data)
				output_resnames_cpu.data[n, atom_idx, :res_name.size(0)].copy_(res_name.data)
				atom_idx += 1
		print(output_atomnames_cpu)
		return output_coords_cpu, output_resnames_cpu, output_atomnames_cpu, torch.tensor(num_atoms, dtype=torch.int, device='cpu')


