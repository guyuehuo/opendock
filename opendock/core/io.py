import os


ALLOWED_CONFIGS_TERMS = ['receptor', 'ligand', 'out',
                         'center_x', 'center_y', 'center_z',
                         'size_x', 'size_y', 'size_z',
                         'threads', 'conformations', 'tasks']


def generate_new_configs(config_inp_fpath: str, 
                         config_out_fpath: str = None):
    """Generate idock style configuration file 
    
    Args:
    ----- 
    config_inp_fpath: str, 
        Input configuration file in Vina style. 
    config_out_fpath: str, 
        Output configuration file in Idock style.
    
    Returns:
    configs: dict, 
        Returned parameters. 
    """
    configs = {}
    with open(config_inp_fpath) as lines:
        for l in lines:
            if len(l.split("=")) == 2:
                key = l.split("=")[0].strip()
                if key in ALLOWED_CONFIGS_TERMS:
                    configs[key] = l.split("=")[1].strip("\n").strip()
                elif key == "cpu":
                    configs['threads'] = l.split("=")[1].strip("\n").strip()
                elif key == "exhaustiveness":
                    configs['tasks'] = int(l.split("=")[1].strip("\n").strip())
                elif key == "num_modes":
                    configs['conformations'] = l.split("=")[1].strip("\n").strip()
    
    if "tasks" not in configs.keys():
        configs["tasks"] = 4

    real_output_dpath = os.path.dirname(configs['out'])
    configs['out'] = real_output_dpath

    if config_out_fpath is not None:
        with open(config_out_fpath, 'w') as tof:
            for key in configs.keys():
                tof.write("{} = {} \n".format(key, configs[key]))
        tof.close()

    return configs


def write_ligand_traj(cnfrs: list, 
                      ligand: None, 
                      output: str, 
                      information: dict = None):
    """
    Write lignad trajectory.

    Args:
    -----
    ligand: LigandConformation, 
    output: str, the trajectory file path.
    information: dict, the information for output if any.
    """

    origin_heavy_atoms_lines = ligand.origin_heavy_atoms_lines
    #print(cnfrs)
    lines = []
    for _idx, cnfr in enumerate(cnfrs):
        # convert cnfr to xyz, coords shape (1, N, 3)
        coord = ligand.cnfr2xyz([cnfr, ])[0]
        lines.append('MODEL%9s' % str(_idx + 1))

        if information is not None:
            for key in list(information.keys()):
                try:
                    lines.append(f"REMARK {key} {information[key][_idx]:.3f}")
                except:
                    lines.append(f"REMARK {key} {information[key][0]:.3f}")

        # make output atom lines
        for num, line in enumerate(origin_heavy_atoms_lines):
            x = coord[num][0].detach().numpy()
            y = coord[num][1].detach().numpy()
            z = coord[num][2].detach().numpy()

            atom_type = line.split()[2]
            pre_element = line.split()[2]
            if pre_element[:2] == "CL":
                element = "Cl"
            elif pre_element[:2] == "BR":
                element = "Br"
            else:
                element = pre_element[0]

            line = "ATOM%7s%5s%4s%2s%4s%12s%8s%8s%6s%6s%12s" % (
                str(num + 1), atom_type, "LIG", "A", "1", "%.3f" % x, "%.3f" % y, "%.3f" % z, "1.00", "0.00", element)
            lines.append(line)
        
        lines.append("TER\nENDMDL")

    with open(output, 'w') as f:
        for line in lines:
            f.writelines(line + '\n')


def write_receptor_traj(cnfrs,
                        receptor: None, 
                        output: str = None,
                        information: dict = None):
    """
    Write the receptor trajectory if considering sidechain flexibility.

    Args:
    ----- 
    receptor: the receptor object. 
    output: str, the output file name.

    1. 2023/3/25: Changed by wzc. Allows coenzymes beginning with HETATM to be read.        

    """
    # obtain the original receptor pdbqt file lines
    rec_original_lines = receptor.receptor_original_lines

    lines = []

    for (idx, cnfr_list) in enumerate(cnfrs):
        lines.append("MODEL%9s" % (str(idx+1)))
        new_rec_ha_xyz = receptor.cnfr2xyz(cnfr_list)
        #num = 0
        if information is not None:
            for key in list(information.keys()):
                try:
                    lines.append(f"REMARK {key} {information[key][_idx]:.3f}")
                except:
                    lines.append(f"REMARK {key} {information[key][0]:.3f}")
        for N, line in enumerate(rec_original_lines):
            # zlz fix: remove the change-line symbol
            line = line.strip("\n")
            ad4_type = line.split()[-1]
            #if ad4_type.endswith("H") or ad4_type.endswith("HD") \
            #    or (not line.startswith("ATOM")): 
            
            if ad4_type.endswith("H") or ad4_type.endswith("HD"):
                continue
            
            #num += 1
            atom_type = line.split()[2]
            if atom_type[:2] == "CL":
                element = "Cl"
            elif atom_type[:2] == "BR":
                element = "Br"
            else:
                element = atom_type[0]

            try:
                #print(N, line)
                if N in receptor.clp_ha_idx:
                    idx = receptor.clp_ha_idx_to_line_num[N]
                    x = new_rec_ha_xyz[idx][0].detach().numpy()
                    y = new_rec_ha_xyz[idx][1].detach().numpy()
                    z = new_rec_ha_xyz[idx][2].detach().numpy()
                    
                    nline = "ATOM%7s%16s%11s%8s%8s%12s%12s" % (
                        str(N+1), line[11:27], "%.3f" % x, 
                        "%.3f" % y, "%.3f" % z, line[54:66], element)
                else:
                    nline = line
                lines.append(nline)
                
            except:
                pass
        lines.append("ENDMDL")

    with open(output, "w") as tf:
        for line in lines:
            tf.write(line + "\n")

