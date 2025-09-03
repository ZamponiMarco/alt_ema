from itertools import product
import json
import os
import pyperclip

import pandas as pd

FOLDER = 'resources/workloads/'

OBJECTIVES = ['overprovisioning', 'underprovisioning']
SHAPES = ['free', 'ramp', 'sawtooth', 'spike']
SHAPE_NAMES = ['Free', 'Ramp', 'Sawtooth', 'Spike']
SHAPE_NAME_MAP = dict(zip(SHAPES, SHAPE_NAMES))

AUTOSCALERS = ['hpa50', 'hpa60', 'step1', 'step2']
AUTOSCALER_NAMES = ['VP50', 'VP60', 'Step1', 'Step2']
AUTOSCALER_NAME_MAP = dict(zip(AUTOSCALERS, AUTOSCALER_NAMES))

def generate_tabular(data):
    """Generate a single tabular environment for the given data."""
    tabular_code = "        \\begin{tabular}{lcccc}\n"
    tabular_code += "            \\toprule\n"
    tabular_code += "            \\emph{Autoscaler} & " + " & ".join(f'\\textbf{{{name}}}' for name in SHAPE_NAMES) + " \\\\\n"
    tabular_code += "            \\midrule\n"
    for index, row in data.iterrows():
        tabular_code += f"            \\textbf{{{AUTOSCALER_NAME_MAP[index]}}} & " + " & ".join(row.values) + " \\\\\n"
    tabular_code += "            \\bottomrule\n"
    tabular_code += "        \\end{tabular}\n"
    return tabular_code

def generate_combined_table(over_data, under_data):
    """Generate the complete table with both objectives side by side."""
    latex_code = "\\begin{table*}[t]\n"
    latex_code += "    \\centering\n"
    latex_code += "    \\textbf{Over-provisioning objective}\\\\[0.5em]\n"
    latex_code += generate_tabular(over_data)
    latex_code += "    \\\\[1em]\n"
    latex_code += "    \\textbf{Under-provisioning objective}\\\\[0.5em]\n"
    latex_code += generate_tabular(under_data)
    latex_code += "    \\vspace{0.5em}\n"
    latex_code += "    \\caption{Summary of test generation outcomes for over-provisioning (top) and under-provisioning (bottom). Cells marked (\\cmark) report a falsifying trace with solving times for the first feasible and the optimal solution (or \\emph{lim} if timed out). Cells marked (\\xmark) indicate problem infeasibility with the corresponding solving time.\n"
    latex_code += "    }\n"
    latex_code += "    \\label{tab:test-generation-outcome}\n"
    latex_code += "\\end{table*}"
    return latex_code

if __name__ == '__main__':
    # Create data structures for both objectives
    data_dict = {}
    
    for objective in OBJECTIVES:
        combinations = list(product(AUTOSCALERS, SHAPES))
        data = pd.DataFrame(index=AUTOSCALERS, columns=SHAPES)
        
        for (a, l) in combinations:
            with open(os.path.join(FOLDER, f'{a}_{objective}_{l}.json')) as f:
                content = json.load(f)
                feas = len(content['users']) > 0
                if feas:
                    first_feas = content['solutions'][0][0]
                    time_str = f"{first_feas:.2f}/lim" if content['status'] == 9 else f"{first_feas:.2f}/{content['time']:.2f}"
                else:
                    time_str = "lim" if content['status'] == 9 else f"{content['time']:.2f}"
                data.at[a, l] = f"\\{'cmark' if feas else 'xmark'} ({time_str})"
                
        data_dict[objective] = data
        
    # Generate the combined table
    latex_table = generate_combined_table(data_dict['overprovisioning'], data_dict['underprovisioning'])
    
    pyperclip.copy(latex_table)
    print("LaTeX table code copied to clipboard.")