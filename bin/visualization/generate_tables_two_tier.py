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
AUTOSCALER_NAMES = ['HPA50', 'HPA60', 'Step1', 'Step2']
AUTOSCALER_NAME_MAP = dict(zip(AUTOSCALERS, AUTOSCALER_NAMES))

def generate_tabular(data):
    """Generate a single tabular environment for the given data."""
    tabular_code = "        \\begin{tabular}{lcccc}\n"
    tabular_code += "            \\toprule\n"
    tabular_code += "            \\emph{Autoscaler} & " + " & ".join(SHAPE_NAMES) + " \\\\\n"
    tabular_code += "            \\midrule\n"
    for index, row in data.iterrows():
        tabular_code += f"            {AUTOSCALER_NAME_MAP[index]} & " + " & ".join(row.values) + " \\\\\n"
    tabular_code += "            \\bottomrule\n"
    tabular_code += "        \\end{tabular}\n"
    return tabular_code

def generate_combined_table(over_data, under_data):
    """Generate the complete table with both objectives side by side."""
    latex_code = "\\begin{table*}[t]\n"
    latex_code += "    \\begin{minipage}{0.5\\linewidth}\n"
    latex_code += "        \\centering\n"
    latex_code += generate_tabular(over_data)
    latex_code += "    \\end{minipage}%\n"
    latex_code += "    \\begin{minipage}{0.5\\linewidth}\n"
    latex_code += "        \\centering\n"
    latex_code += generate_tabular(under_data)
    latex_code += "    \\end{minipage}\n"
    latex_code += "    \\vspace{0.5em}\n"
    latex_code += "    \\caption{Summary of test generation outcomes for the over-provisioning (left) and under-provisioning (right) objectives. Each cell reports whether a falsifying trace was found (\\cmark) or not (\\xmark), followed by the solving time of the falsification problem in seconds (in parentheses), or ``lim'' if the time limit was reached. Note that results marked as \\xmark and ``lim'' indicate an inconclusive outcome, whereas all other cases represent a (possibly sub-optimal) conclusive result.\n"
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
                time_str = "lim" if content['status'] == 9 else f"{content['time']:.2f}"
                data.at[a, l] = f"\\{'cmark' if feas else 'xmark'} ({time_str})"
                
        data_dict[objective] = data
        
    # Generate the combined table
    latex_table = generate_combined_table(data_dict['overprovisioning'], data_dict['underprovisioning'])
    
    pyperclip.copy(latex_table)
    print("LaTeX table code copied to clipboard.")