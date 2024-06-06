import re
import numpy as np
import pymatgen.core as p
from cation_electronegativity import A_site_cation_electronegativity, B_site_cation_electronegativity
import time
import ast
import pandas as pd
import pickle

# return [[elements],[counts]]
def get_elements_for_pymatgen(expression):
    element_pat = re.compile("([A-Z][a-z]*)(\d*[.]*\d*)")
    elements = [[],[]]
    for (element_name, count) in element_pat.findall(expression):
        elements[0].append(element_name)
        if count == "":
            count = 1.0
        else:
            count = float(count)
        elements[1].append(count)
    return elements

# return AxByOz, multiply 
# multiply means coverting decimal value to none-decimal
def get_simplified_comp(A_ion, B_ion, guess_O_no=3,multiply=1000):
    # use simplified for oxi_state_guesses reducing computational effort
    comp_simplified = ""
    # greatest common divisor
    nums = A_ion[1]+B_ion[1]+[guess_O_no]
    nums = [int(round(float(i),2)*multiply) for i in nums]
    gcd = np.gcd.reduce(nums)
    for i in range(len(A_ion[0])):
        comp_simplified += str(A_ion[0][i]) + str(int(multiply*float(A_ion[1][i])/gcd))
    for i in range(len(B_ion[0])):
        comp_simplified += str(B_ion[0][i]) + str(int(multiply*float(B_ion[1][i])/gcd))
    comp_simplified += "O" + str(int(int(round(guess_O_no,2)*multiply)/gcd))  
    multiply = multiply/gcd
    return comp_simplified, multiply

# return [ 
#   {'La': 3.0, 'Sr': 2.0, 'Fe': 3.5, 'O': -2.0} 
# ]
# return None for not able to guess
def guess_oxi_state(comp_simplified):#np.arange(2.5,3.1,0.1)):
    print("guess exp: ",comp_simplified)
    comp = p.Composition(comp_simplified)
    oxi_dict = {}
    for key in A_site_cation_electronegativity.keys():
        value = list(A_site_cation_electronegativity[key].keys())
        oxi_dict[key] = value
    oxi_dict["O"] = [-2]
    oxi_state = comp.oxi_state_guesses(oxi_states_override=oxi_dict,all_oxi_states=True,max_sites=-1)
    if len(oxi_state) == 0:
        print("No possible oxi state guess")
        return []
    else:
        # get the most-likely state
        possible_oxi_state = []
        for oxi_sta in oxi_state:
            abolish = False
            for element, oxi in oxi_sta.items():
                if element not in oxi_dict.keys():
                    # print(element, oxi)
                    common_oxi = B_site_cation_electronegativity[element].keys()
                    # print(common_oxi)
                    if oxi not in common_oxi:
                        if element not in ["Co","Ni","Fe"]:
                            abolish = True
            if not abolish:
                possible_oxi_state.append(oxi_sta)
        # print(possible_oxi_state)
        
        permutations_dicts = []
        for oxi_state in possible_oxi_state:
            for element, oxi in oxi_state.items():
                # A_site
                if element in oxi_dict.keys():
                    continue
                # B_site
                else:
                    common_oxi = B_site_cation_electronegativity[element].keys()
                    common_oxi = list(common_oxi)
                    # print(common_oxi)
                    # solve equation
                    if oxi not in common_oxi:
                        oxi_state[element] = []
                        # only those metals are listed in orignal data with multipul oxi_state in one metarial
                        if element not in ["Mn","Co","Ni","Fe"]:
                            print("1")
                            continue
                        elif len(common_oxi) == 2:
                            ratio = np.linalg.solve([common_oxi,[1,1]], [oxi,1])
                            if ratio[0] >= 0 and ratio[1] >= 0:
                                oxi_state[element].append({common_oxi[0]: ratio[0], common_oxi[1]: ratio[1]})
                            else:
                                continue

                        elif len(common_oxi) == 3:
                            possible_combination = []
                            possible_1st_ratio = np.arange(0,1.01,0.01)
                            possible_1st_ratio = np.append(possible_1st_ratio,[0.0793,0.1123,0.1252,0.1541,0.2535,0.345])
                            for first_ratio in possible_1st_ratio:
                                ratio = np.linalg.solve([common_oxi[1:],[1,1]], [oxi-first_ratio*common_oxi[0],1-first_ratio])
                                if ratio[0] >= 0 and ratio[1] >= 0:
                                    possible_combination.append([first_ratio,ratio[0],ratio[1]])
                            if len(possible_combination) <= 0:
                                continue
                            else:
                                for comb in possible_combination:
                                    temp = {}
                                    for oxi,i in zip(common_oxi,comb):
                                        temp[int(oxi)] = i
                                    oxi_state[element].append(temp)
                        else:
                            print("4")
                            continue
                    else:
                        continue

                    
            temp = {}
            empty= False
            for element, oxi in oxi_state.items():  
                if isinstance(oxi, float):
                    temp[element] = [{oxi: 1.0}]
                else:
                    if len(oxi) == 0:
                        empty = True
                    else:
                        temp[element] = oxi
            if not empty:
                import itertools
                keys, values = zip(*temp.items())
                permutations_dicts = permutations_dicts + [dict(zip(keys, v)) for v in itertools.product(*values)]

        if len(permutations_dicts) < 1:
            print("No possible oxi state guess")
            return []
        else:
            return permutations_dicts

def get_feature_value(expression,A_ion,B_ion,oxi_state,multiply=100,guess_O_no=3.0):
    comp = p.Composition(expression)
    
    # Formula weight 
    weight = comp.weight/multiply

    # weight for computing average
    oxi_As = {ele: oxi_state[ele] for ele in oxi_state.keys() if ele in A_ion[0]}
    oxi_Bs = {ele: oxi_state[ele] for ele in oxi_state.keys() if ele in B_ion[0]}

    # average cation valence 
    V_A = {}
    for k,v in oxi_As.items():
        V_A[k] = np.sum([k_*v_ for k_,v_ in v.items()])
    V_A = np.sum([A_ion[1][A_ion[0].index(k)]*v for k,v in V_A.items()])
    V_B = {}
    for k,v in oxi_Bs.items():
        V_B[k] = np.sum([k_*v_ for k_,v_ in v.items()])
    V_B = np.sum([float(B_ion[1][B_ion[0].index(k)])*float(v) for k,v in V_B.items()])

    # average  cation radius
    R_A = {}
    for k,v in oxi_As.items():
        try:
            R_A[k] = np.sum([p.Species(k,oxidation_state=abs(k_)).get_shannon_radius(cn="XII",spin="",radius_type="ionic")*v_ for k_,v_ in v.items()])
        except:
            if k == "Pr":
                R_A["Pr"] = 1.3100
            elif k in ["Bi","Y","Sn"]:
                R_A[k] = np.sum([p.Species(k,oxidation_state=abs(k_)).get_shannon_radius(cn="VI",spin="",radius_type="ionic")*v_ for k_,v_ in v.items()])
            else:
                print("%s radius error in R_A"%k)
                R_A[k] = np.sum([p.Species(k,oxidation_state=abs(k_)).get_shannon_radius(cn="XII",spin="",radius_type="ionic")*v_ for k_,v_ in v.items()])
    R_A = np.sum([A_ion[1][A_ion[0].index(k)]*v for k,v in R_A.items()])

    R_B = {}
    for k,v in oxi_Bs.items():
        try:
            R_B[k] = np.sum([p.Species(k,oxidation_state=abs(k_)).get_shannon_radius(cn="VI",spin="Low Spin",radius_type="ionic")*v_ for k_,v_ in v.items()])
        except:
            print("%s radius error in R_B"%k)
            R_B[k] = np.sum([p.Species(k,oxidation_state=abs(k_)).get_shannon_radius(cn="VI",spin="",radius_type="ionic")*v_ for k_,v_ in v.items()])
    R_B = np.sum([B_ion[1][B_ion[0].index(k)]*v for k,v in R_B.items()])

    # Average cation electronegativity 
    X_A = {}
    for k,v in oxi_As.items():
        X_A[k] = np.sum([A_site_cation_electronegativity[k][k_]*v_ for k_,v_ in v.items()])
    X_A = np.sum([A_ion[1][A_ion[0].index(k)]*v for k,v in X_A.items()])

    X_B = {}
    for k,v in oxi_Bs.items():
        X_B[k] = np.sum([B_site_cation_electronegativity[k][k_]*v_ for k_,v_ in v.items()])
    X_B = np.sum([B_ion[1][B_ion[0].index(k)]*v for k,v in X_B.items()])

    # ionization energies
    # List of ionization energies. First value is the first ionization energy, second is the second ionization energy, etc. Note that this is zero-based indexing! So Element.ionization_energies[0] refer to the 1st ionization energy. Values are from the NIST Atomic Spectra Database. Missing values are None.
    # For a M cation with N+ oxidation state, its ionization energy denotes the Nth ionization energy of M atom in this work.
    AIE = {}
    for k,v in oxi_As.items():
        AIE[k] = np.sum([p.Species(k,oxidation_state=abs(k_)).ionization_energies[int(abs(k_))-1]*v_ for k_,v_ in v.items()])
    AIE = np.sum([A_ion[1][A_ion[0].index(k)]*v for k,v in AIE.items()])

    BIE = {}
    for k,v in oxi_Bs.items():
        BIE[k] = np.sum([p.Species(k,oxidation_state=abs(k_)).ionization_energies[int(abs(k_))-1]*v_ for k_,v_ in v.items()])
    BIE = np.sum([B_ion[1][B_ion[0].index(k)]*v for k,v in BIE.items()])

    # weighted_RO normalized by 3
    R_O = 1.4*guess_O_no/3
    # t(tolerance factor)
    t = (R_A + R_O)/(np.sqrt(2) * (R_B + R_O))
    t_ = (R_A + 1.4)/(np.sqrt(2) * (R_B + 1.4))
    # μ(Octahedral_factor)
    u = R_B / R_O
    u_ = R_B / 1.4

    # entropy
    E_A = np.sum([-i*np.log(i) for i in A_ion[1]])
    E_B = np.sum([-i*np.log(i) for i in B_ion[1]])

    E = np.sum([-(i/(2.0+guess_O_no))*np.log(i/(2.0+guess_O_no)) for i in A_ion[1]]) + np.sum([-(i/(2.0+guess_O_no))*np.log(i/(2.0+guess_O_no)) for i in B_ion[1]]) + (-guess_O_no/(2.0+guess_O_no)*np.log(guess_O_no/(2.0+guess_O_no)))

    return weight, round(V_A,2), round(V_B,2), round(V_A+V_B,2), R_A, R_B, R_A+R_B, R_O, X_A, X_B, X_A+X_B, AIE, BIE, t, t_, u, u_, E_A, E_B, E, {**oxi_As,**oxi_Bs}

def read_oxi_state(valence,B_ion):
    valence_prop_pat = re.compile("([A-Z][a-z]*)([^;]+)(;)")
    prop_pat = re.compile("(\d)(:)(\d*[.]*\d*)")

    valence_prop_dict = {}
    for element, valence_prop, _ in valence_prop_pat.findall(valence):
        valence_prop_dict[element] = {}
        for oxi_number, _, prop in prop_pat.findall(valence_prop):
            valence_prop_dict[element][float(oxi_number)] = float(prop)#*B_ion[1]
    return valence_prop_dict

def encode(file_name,predict=False,file_path_head="./",vsum_index=6,guess_O_range=np.arange(2.0,3.1,0.1)):
    with open("%s%s.xlsx"%(file_path_head,file_name),'rb') as f:
        if predict:
            original_data = pd.read_excel(f,header=[0])
        else:
            original_data = pd.read_excel(f,header=[0,1,2])

        if predict:
            feature_names = ["Perovskite composition(ABO3)", "A_site", "B_site", "weight", "V_A", "V_B", "V_(A+B)", "R_A", "R_B", "R_(A+B)", "R_O", "XA", "XB", "X_(A+B)", "AIE (A-site)", "BIE (B-site)", "t (tolerance factor)", "t (O=1.4)", "μ (Octahedral_factor)", "μ (O=1.4)", "E_A", "E_B", "E", "Valence: proportion", ]
        else:
            feature_names = ["Perovskite composition(ABO3)", "A_site", "B_site", "weight", "V_A", "V_B", "V_(A+B)", "R_A", "R_B", "R_(A+B)", "R_O", "XA", "XB", "X_(A+B)", "AIE (A-site)", "BIE (B-site)", "t (tolerance factor)", "t (O=1.4)", "μ (Octahedral_factor)", "μ (O=1.4)", "E_A", "E_B", "E", "Mass loading (mg cm-2)", "Electrolyte (M KOH)", "overpotential (mV)", "Tafel slope (mV Dec-1)", "Substrate", "Valence: proportion", "year", "Ref"]

        x_all = original_data.values

        x_encoded = []
        slice_count = 0
        for x in x_all:
            guess_O_no_all = []
            if predict:
                guess_O_no_all = guess_O_range
            else:
                guess_O_no = x[vsum_index]/2.0
                guess_O_no_all.append(guess_O_no)

            current_time = time.time()

            for guess_O_no in guess_O_no_all:
                A_ion = get_elements_for_pymatgen(x[1])
                B_ion = get_elements_for_pymatgen(x[2])
                comp_simplified, multiply = get_simplified_comp(A_ion,B_ion,guess_O_no=round(guess_O_no,2))
                if predict:
                    current_time = time.time()  
                    oxi_state = guess_oxi_state(comp_simplified)
                    if (time.time()-current_time)>5:
                        print("%s time used: %.4f"%(x[0],time.time()-current_time))
                else:
                    oxi_state = eval(x[-3])
                    oxi_state = [oxi_state]

                if len(oxi_state) > 0:
                    for oxi_sta in oxi_state:
                        vector = get_feature_value(comp_simplified,A_ion,B_ion,oxi_sta,multiply=multiply,guess_O_no=guess_O_no)
                        if vector == None:
                            continue
                        else:

                            temp = [x[0],x[1],x[2]]
                            for i in vector[:-1]:
                                temp.append("%.4f"%i)
                            if predict:
                                temp.append(oxi_sta)
                            else:
                                for i in x[-8:]:
                                    temp.append(i)
                            x_encoded.append(temp)

                    if len(x_encoded) >= 100000:
                        df = pd.DataFrame(x_encoded, columns=feature_names)
                        df.to_excel(f"{file_path_head}cation_{file_name}.slice {slice_count}.xlsx", index=False)
                        print(f"Saved data to {file_path_head}cation_{file_name}.slice {slice_count}.xlsx")
                        x_encoded = []
                        slice_count += 1

        if len(x_encoded) > 0:
            df = pd.DataFrame(x_encoded, columns=feature_names)
            df.to_excel(f"{file_path_head}cation_{file_name}.slice {slice_count}.xlsx", index=False)
            print(f"Saved data to {file_path_head}cation_{file_name}.slice {slice_count}.xlsx")                    
