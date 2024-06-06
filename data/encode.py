import re
import numpy as np
import pymatgen.core as p
from cation_electronegativity import A_site_cation_electronegativity, B_site_cation_electronegativity
import time
import ast
import pandas as pd
import pickle
import itertools

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
def get_simplified_comp(A_ion, B_ion, guess_O_no=3,multiply=10000):
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

class oxi_state_guessor():
    def __init__(self,mixed_elements=["Co","Fe"]):
        self.oxi_dict = {}
        self.oxi_dict["O"] = [{-2.0: 1.0}]
        for key in A_site_cation_electronegativity.keys():
            value = list(A_site_cation_electronegativity[key].keys())
            self.oxi_dict[key] = [{float(i): 1.0}for i in value]
        for key in B_site_cation_electronegativity.keys():
            if key not in mixed_elements:
                value = list(B_site_cation_electronegativity[key].keys())
                self.oxi_dict[key] = [{float(i): 1.0}for i in value]

        for element in mixed_elements:
            with open("./oxi_states_%s.xlsx"%(element),"rb") as f:
                self.oxi_dict[element] = []
                oxi_file = pd.read_excel(f)
                oxi_header = oxi_file.columns[:-1]
                oxi_file = oxi_file.values
                for value in oxi_file:
                    temp = {}
                    for valence,percent in zip(oxi_header,value):
                        temp[float(valence)] = percent
                    self.oxi_dict[element].append(temp)

    def predict(self,comp_simplified):
        print("guess exp: ",comp_simplified)
        elements = get_elements_for_pymatgen(comp_simplified)
        combinations = []
        for ele in elements[0]:
            combinations.append(self.oxi_dict[ele])
        combinations_all = list(itertools.product(*combinations))
        combinations_balance = []
        for comb in combinations_all:
            comb = list(comb)
            valence = 0
            for comb_,percent in zip(comb,elements[1]):
                for key in comb_.keys():
                    valence += key*comb_[key]*percent
            if valence == 0:
                temp = {}
                for ele,val in zip(elements[0],comb):
                    temp[ele] = val
                combinations_balance.append(temp)
        return combinations_balance

def get_feature_value(expression,A_ion,B_ion,oxi_state,multiply=100,guess_O_no=3.0):
    comp = p.Composition(expression)
    
    # Formula weight 
    weight = comp.weight/multiply

    # weight for computing average
    oxi_As = {ele: oxi_state[ele] for ele in oxi_state.keys() if ele in A_ion[0]}
    oxi_Bs = {ele: oxi_state[ele] for ele in oxi_state.keys() if ele in B_ion[0]}
    # print(oxi_As,oxi_Bs)

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

def encode(file_name, predict=False, file_path_head="./", vsum_index=7, oxi_state_index=-3, guess_O_range=np.arange(2.0,3.1,0.1),guessor=oxi_state_guessor()):
    with open("%s%s.xlsx"%(file_path_head, file_name),'rb') as f:
        if predict:
            original_data = pd.read_excel(f, header=[0])
        else:
            original_data = pd.read_excel(f, header=[0,1,2])

        if predict:
            feature_names = ["Perovskite composition(ABO3)", "A_site", "B_site", "weight", "V_A", "V_B", "V_(A+B)", "R_A", "R_B", "R_(A+B)", "R_O", "XA", "XB", "X_(A+B)", "AIE (A-site)", "BIE (B-site)", "t (tolerance factor)", "t (O=1.4)", "u (Octahedral_factor)", "u (O=1.4)", "E_A", "E_B", "E", "Valence: proportion", ]
        else:
            feature_names = ["Perovskite composition(ABO3)", "A_site", "B_site", "weight", "V_A", "V_B", "V_(A+B)", "R_A", "R_B", "R_(A+B)", "R_O", "XA", "XB", "X_(A+B)", "AIE (A-site)", "BIE (B-site)", "t (tolerance factor)", "t (O=1.4)", "u (Octahedral_factor)", "u (O=1.4)", "E_A", "E_B", "E", "Mass loading (mg cm-2)", "Electrolyte (M KOH)", "overpotential (mV)", "Tafel slope (mV Dec-1)", "Substrate", "Valence: proportion", "year", "Ref"]

        x_all = original_data.values

        x_encoded = []
        slice_count = 0
        for x in x_all:
            print(x[0])
            guess_O_no_all = []
            if predict:
                guess_O_no_all = guess_O_range
            else:
                guess_O_no = x[vsum_index]/2.0
                guess_O_no_all.append(guess_O_no)

            for guess_O_no in guess_O_no_all:
                A_ion = get_elements_for_pymatgen(x[1])
                B_ion = get_elements_for_pymatgen(x[2])
                comp_simplified, multiply = get_simplified_comp(A_ion, B_ion, guess_O_no=round(guess_O_no,4))
                if predict:
                    current_time = time.time()  
                    oxi_state = guessor.predict(comp_simplified)
                    if (time.time()-current_time)>5:
                        print("%s time used: %.4f"%(x[0],time.time()-current_time))
                else:
                    oxi_state = [eval(x[oxi_state_index])]

                if len(oxi_state) > 0:
                    for oxi_sta in oxi_state:
                        vector = get_feature_value(comp_simplified, A_ion, B_ion, oxi_sta, multiply=multiply, guess_O_no=guess_O_no)
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
