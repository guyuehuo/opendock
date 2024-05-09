
import os, sys 
import shutil
import argparse
import subprocess as sp 
#from openbabel import openbabel as ob
import openbabel as ob
from opendock.core.ligand import Ligand


def convert_mol(inp, out):
     
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats(inp.split('.')[-1], "pdbqt")
    ligand = ob.OBMol()
    obConversion.ReadFile(ligand, inp)
    obConversion.WriteFile(ligand, out)

def run_cmd(cmd):

    print("Running cmd ", cmd)
    job = sp.Popen(cmd, shell=True)
    job.communicate()

def write_config(fpath, rec, lig, out, center, size=15, ntasks=16):

    content = f"""
receptor = {rec}
ligand = {lig}
out = {out}/output_clusters.pdbqt
size_x = {size}
size_y = {size}
size_z = {size}
center_x = {center[0]:.3f}
center_y = {center[1]:.3f}
center_z = {center[2]:.3f}
exhaustiveness = {ntasks}
"""

    with open(fpath, 'w') as tofile:
        tofile.write(content)
 
def arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--receptor", default="receptor.pdb", help="Input receptor file.")
    parser.add_argument("-l", "--ligand", 
                        default="ligand.mol2", help="Input ligand file.")
    parser.add_argument("-ref", "--reference", default="ligand_ref.mol2", help="Reference ligand file.")
    parser.add_argument("-o", "--output", default="output", help="Output directory.")
    parser.add_argument("-bs", "--boxsize", default=15, help="Docking pocket size.")
    parser.add_argument("-n", "--ntasks", default=16, help="Exhaustiveness.")

    args = parser.parse_args()
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    
    return args 


if __name__ == "__main__":

    args = arguments()

    # make receptor file 
    os.makedirs(args.output, exist_ok=True)
    rec_fpath = os.path.abspath(os.path.join(args.output, "receptor.pdbqt"))
    if args.receptor.split(".")[-1] == "pdbqt":
        shutil.copy(args.receptor, rec_fpath)
    else:
        cmd = f"prepare_receptor4.py -r {args.receptor} -o {rec_fpath}"
        try:
            run_cmd(cmd) 
        except:
            pass

    if not os.path.exists(rec_fpath):
        convert_mol(args.receptor, rec_fpath)

    # make ligand file
    lig_fpath = os.path.abspath(os.path.join(args.output, "ligand.pdbqt"))
    if not os.path.exists(lig_fpath):
        convert_mol(args.ligand, lig_fpath)

    # make reference file
    ref_fpath = os.path.join(args.output, "refer.pdbqt")
    if not os.path.exists(ref_fpath):
        convert_mol(args.reference, ref_fpath)
    
    # pocket center
    lig = Ligand(ref_fpath)
    lig.parse_ligand()
    box_center = lig.init_lig_heavy_atoms_xyz.detach().numpy().reshape((-1, 3)).mean(axis=0)

    write_config(os.path.join(args.output, "configs.txt"), rec_fpath, 
                 lig_fpath, os.path.abspath(args.output), box_center, args.boxsize, args.ntasks)
