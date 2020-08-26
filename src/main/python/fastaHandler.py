import os
import pickle
from dataset import *

class FASTAHandler(object):
    def __init__(self):
        self.filePath = ""
        self.lines = []
        self.sequenceCache = {}

    def load(self, path):
        """
        Load a file for handling later
        :param path: the path to the file to be loaded
        :return: None
        """

        self.filePath = path
        with open(path) as f:
            self.lines = f.readlines()
        self.lines.append(">")
        self.sequenceCache = {}
        self.accessionCache = {}

    def accessionList(self):
        accessions = []
        for i, line in enumerate(self.lines):
            if ">EPI_" in line: # is a GISAID file
                tokens = line.split(".")
                accession = tokens[0][1:]
                accessions.append(accession)
                self.accessionCache[accession] = i
            elif ">" in line and line != ">": # is an NCBI file
                accession = line[1:9]
                accessions.append(accession)
                self.accessionCache[accession] = i
        return accessions

    def loadSequence(self, accession):
        """
        Load a sequence from it's accession ID
        :param accession: the ID of the sequence
        :return: the loaded sequence
        """

        if accession in self.sequenceCache:
            return self.sequenceCache.get(accession)
        else:
            sequence = ""
            reading = False
            hasLoaded = False
            need_proc = self.lines[self.accessionCache.get(accession, 0):]
            for line in need_proc:
                if line.startswith(">") and hasLoaded:
                    break
                if reading:
                    sequence += line
                if accession in line: # found the sequence, start reading
                    reading = True
                    hasLoaded = True
            if sequence == "":
                raise Exception(f"sequence {accession} not found in {self.path}")
            else:
                self.sequenceCache[accession] = sequence
                return clean(sequence)

def clean(sequence):
    """
    Remove newlines from a sequence and make it uppercase
    :param sequence: the sequence to be cleaned
    :return: cleaned sequence
    """

    sequence = sequence.replace("\n", "").upper()
    assert not "\n" in sequence
    spaces = abs(566 - len(sequence)) * " "
    return sequence + spaces

def translateToSeq(numerical):
    seq = ''
    for decimal in numerical:
        rounded = round(decimal * 3)/3
        reversed_dictionary = dict(map(reversed, AMINO_ACIDS.items()))
        seq += reversed_dictionary[rounded]
    return seq

def translateToNumerical(seq):
    return [AMINO_ACIDS[x] for x in seq]

def getZeros(number):
    return (2-len(str(i)))*"0"

AMINO_ACIDS = {}
with open("/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python/amino_acids.txt", 'r') as f:
    txt = f.readlines()
for i, acid in enumerate(txt[0]):
    AMINO_ACIDS[acid] = i + 2 # 0 and 1 are PAD and EOSnce(handler.accessionList()[0]))