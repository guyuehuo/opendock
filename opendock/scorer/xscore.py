#!/usr/bin/env python

import pandas as pd
import numpy as np
import subprocess as sp
import shutil
import os,sys
import uuid
from opendock.scorer.RTMScore.utils import obabel
from opendock.scorer.scoring_function import ExternalScoringFunction


XSCORE_DPATH = "/user/zhengliangzhen/apps/xscore_v1.3_zlz"


class Xscore(object):

    def __init__(self, receptor_fpath, ligand_fpath, cofactor=None, xscore_root=XSCORE_DPATH):
        self.pdbfn = receptor_fpath
        self.ligfn = ligand_fpath
        self.cof = cofactor
        self.xscore_root = xscore_root
        self.cleanup = True

    def _prep_input(self, xscore_inp, xscore_table, xscore_log="xscore.log"):
        xscore_input = """
######################################################################
#                            XTOOL/SCORE                             # 
######################################################################
###
FUNCTION        SCORE
###
### specify the input and output files -------------------------------
###
#
RECEPTOR_PDB_FILE    %s
REFERENCE_MOL2_FILE  none
COFACTOR_MOL2_FILE   none
LIGAND_MOL2_FILE     %s
#
OUTPUT_TABLE_FILE    %s
OUTPUT_LOG_FILE      %s
###
### how many top hits to extract from the LIGAND_MOL2_FILE?
###
NUMBER_OF_HITS       5 
HITS_DIRECTORY       ./hits
###
### want to include atomic binding scores in the resulting Mol2 files?
###
SHOW_ATOM_BIND_SCORE    YES              [YES/NO]
###
### set up scoring functions -----------------------------------------
###
APPLY_HPSCORE         YES               [YES/NO]
        HPSCORE_CVDW  0.004 
        HPSCORE_CHB   0.054
        HPSCORE_CHP   0.009
        HPSCORE_CRT  -0.061
        HPSCORE_C0    3.441
APPLY_HMSCORE         YES               [YES/NO]
        HMSCORE_CVDW  0.004
        HMSCORE_CHB   0.101
        HMSCORE_CHM   0.387
        HMSCORE_CRT  -0.097
        HMSCORE_C0    3.567
APPLY_HSSCORE         YES               [YES/NO]
        HSSCORE_CVDW  0.004
        HSSCORE_CHB   0.073
        HSSCORE_CHS   0.004
        HSSCORE_CRT  -0.090
        HSSCORE_C0    3.328
###
### set up chemical rules for pre-screening ligand molecules ---------
###
APPLY_CHEMICAL_RULES    NO            [YES/NO]
        MAXIMAL_MOLECULAR_WEIGHT      2000.0
        MINIMAL_MOLECULAR_WEIGHT      10.0
        MAXIMAL_LOGP                  15.00
        MINIMAL_LOGP                  -10.00
        MAXIMAL_HB_ATOM               100 
        MINIMAL_HB_ATOM               0 
###
###
        """ % (self.pdbfn, self.ligfn, xscore_table, xscore_log)

        with open(xscore_inp, 'w') as tofile:
            tofile.write(xscore_input)

        return self

    def _parse_log(self, log_file):
        if not os.path.exists(log_file):
            print("[WARNING] xscore log file out found ...")
            return 99.99

        with open(log_file) as lines:
            log_lines = [ x for x in lines]

        ligand_code = "ligand"
        if log_lines is None or len(log_lines) < 4:
            print("[WARNING] empty xscore log file ...")
            return 99.99

        '''
        start_ndx, end_ndx = 0, 0
        for j, l in enumerate(log_lines):
            if l.startswith("ID"):
                start_ndx = j + 2
            elif l.startswith("Total"):
                end_ndx = j - 2
            else:
                pass

        try:
            scores = [log_lines[x].split() for x in range(start_ndx, end_ndx+1)]
        except IndexError:
            print("[Warning] did not parse scores file well, exit now...")
            print("LOG FILES: ", log_lines)
            return 99.99

        if len(scores) <= 1:
            print("[Warning] xscore parse decomposed energy file error!!!")
            return 99.99'''
        try:
            scores = [float(x.split()[-1]) for x in log_lines \
                      if (len(x.split()) > 7) and "Total" in x]
            return scores[0] * -1.0
        except:
            print("[WARNING] parse xscore log file error...")
            return 99.99

    def _run_xscore(self, xscore_root, input_file):
        # setup necessary environment path
        os.environ["XTOOL_HOME"] = xscore_root
        os.environ["XTOOL_PARAMETER"] = os.path.join(xscore_root, "parameter")
        os.environ["XSCORE_PARAMETER"] = os.path.join(xscore_root, "parameter")
        os.environ["XTOOL_BIN"] = os.path.join(xscore_root, "bin")

        cmd = f"$XTOOL_BIN/xscore_64bit {input_file} > /dev/null 2>&1" 
        #print(f"Running cmd: {cmd}")

        try:
            job = sp.Popen(cmd, shell=True)
            job.communicate()
        except:
            print("Something run with xscore, retry!!!")
            self._copy_paramters()
            job = sp.Popen(cmd, shell=True)
            job.communicate()

        return self

    def run_xscore(self):
        unique_name = str(uuid.uuid4().hex)[:8]

        self._prep_input("xscore_{}.inp".format(unique_name),
                         "xscore_{}.table".format(unique_name),
                         "xscore_{}.log".format(unique_name))
        
        self._run_xscore(input_file="xscore_{}.inp".format(unique_name), 
                         xscore_root=self.xscore_root)
        
        try:
            score = self._parse_log("xscore_{}.log".format(unique_name))
        except:
            score = 99.99

        if self.cleanup:
            try:
                os.remove("xscore_{}.inp".format(unique_name))
            except:
                pass
                
            try:
                os.remove("xscore_{}.table".format(unique_name))
            except:
                pass

            try:
                os.remove("xscore_{}.log".format(unique_name))
            except:
                pass

            try:
                shutil.rmtree("hits")
            except:
                pass

        return [score, ]

    def _copy_paramters(self):

        try:
            pwd = os.getcwd()
            param_dir = os.path.join(pwd, "../parameters")
            os.mkdir(param_dir)

            shutil.copytree(os.path.join(self.xscore_root, "parameter"), param_dir)
        except:
            pass

        return self


class XscoreSF(ExternalScoringFunction):

    def __init__(self, receptor = None, ligand = None, **kwargs):
        super(XscoreSF, self).__init__(receptor=receptor, ligand=ligand)

        self.receptor = receptor
        self.ligand = ligand

        self.tmp_dpath = None
        self.receptor_fpath = None
        self.ligand_fpath   = None
        self.verbose = kwargs.pop('verbose', False)
    
    def _score(self, receptor_fpath=None, ligand_fpath=None):
        print(self.tmp_dpath)
        # convert protein
        #obabel(receptor_fpath, f'{self.tmp_dpath}/receptor.pdb')

        # convert ligand
        obabel(ligand_fpath, f'{self.tmp_dpath}/docked_ligands.mol2')

        # xscore
        xscore = Xscore(receptor_fpath, 
                        f'{self.tmp_dpath}/docked_ligands.mol2', 
                        xscore_root=XSCORE_DPATH)
        scores = xscore.run_xscore()

        return scores


if __name__ == "__main__":

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation

    if len(sys.argv) < 2:
        print("usage: xscore.py protein.pdbqt ligand.pdbqt output tag")
        sys.exit(0)
    
    # ligand center
    ligand = LigandConformation(sys.argv[1])
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center) 

    # define a flexible ligand object 
    receptor = ReceptorConformation(sys.argv[2], 
                                    xyz_center,
                                    ligand.init_heavy_atoms_coords)
    tf = open(sys.argv[3], 'w')
    try:
        tag = sys.argv[4]
    except:
        tag = "decoy"

    sf = XscoreSF(receptor=receptor, ligand=ligand, verbose=True)
    #print(sf.scoring())
    score = sf.scoring(remove_temp=False) 
    score = score.detach().numpy().ravel()[0] 
    tf.write(f'{tag},{score}\n')
    tf.close()
