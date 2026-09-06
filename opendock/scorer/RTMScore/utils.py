
try:
    from openbabel import openbabel as ob
except:
    import openbabel as ob
    
import os


def obabel(infile, outfile):
    basename = os.path.basename(infile).split(".")[0]
    _format = outfile.split(".")[-1]

    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats(basename, _format)
    mol = ob.OBMol()
    obConversion.ReadFile(mol, infile)
    obConversion.WriteFile(mol, outfile)

    