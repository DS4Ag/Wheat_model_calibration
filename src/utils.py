import pandas as pd
import os
import re
import yaml

def validate_file_path(file_path):
    """
    Validates that the input file path exists.
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError(f"The file path must be a non-empty string.")
    if not os.path.isfile(file_path):
        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_path)
        raise FileNotFoundError(f"The file '{file_name}' does not exist at: {file_dir}")
    return file_path


def simulations_lines(file_path):
    """
    Identifies and extracts the line ranges associated with specific treatments in the OVERVIEW output file.

    Parameters:
    file_path (str): The path to the text file containing tOVERVIEW output file.

    Returns:
    dict: A dictionary where the keys are TREATMENT names and the values are
          tuples containing the start and end line numbers.
    """

    # Initialize dictionaries and lists to store relevant data
    result_dict = {}
    dssat_lines = []
    run_lines = []

    # Open the file and read all lines into a list
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterate through each line to identify lines starting with '*DSSAT' and '*RUN'
    for i, line in enumerate(lines):
        if '*DSSAT' in line:
            # Store the line number of each '*DSSAT' occurrence
            dssat_lines.append(i)
        if line.strip().startswith('TREATMENT'):
            # Store the line number and the content of each 'TREATMENT' line
            run_lines.append((i, line.strip()))

    # Process the stored lines to populate the result dictionary
    for i, start_line in enumerate(dssat_lines):
        if i < len(dssat_lines) - 1:
            # Determine the end line for the current '*DSSAT' section
            end_line = dssat_lines[i + 1] - 1
        else:
            # If it's the last '*DSSAT' section, set the end line as the last line of the file
            end_line = len(lines) - 1
            end_line = len(lines)

        # Find the appropriate '*RUN' line within the current '*DSSAT' section
        run_info = None
        for run_start, run_line in run_lines:
            if run_start >= start_line and run_start <= end_line:
                # Extract only the 'treatment' name from the 'TREATMENT -n' line
                run_info = run_line.split(':')[1].strip().rsplit(maxsplit=1)[0]

                break

        # If the treatment information was found, store it in the result dictionary
        if run_info:
            result_dict[run_info] = (start_line, end_line)

    # Return the entire dictionary with all treatment information
    return result_dict


def extract_simulation_data(file_path):
    """
    Extracts simulation data for each cultivar, including the experiment information,
    and returns a DataFrame with all the data.

    Parameters:
    file_path (str): The path to the text file containing the growth aspects data.

    Returns:
    pd.DataFrame: A DataFrame containing the parsed data for all cultivars, including the experiment info.
    """
    # Get the dictionary with the line ranges for each cultivar
    treatment_dict = simulations_lines(file_path)

    # Initialize an empty DataFrame to store all the data
    all_data = pd.DataFrame(
        columns=['TREATMENT', 'cultivar', 'VARIABLE', 'VALUE_SIMULATED', 'VALUE_MEASURED', 'EXPERIMENT', 'POSITION'])

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Iterate through each cultivar and extract data
        for treatment, (start_line, end_line) in treatment_dict.items():
            cultivar_data = []
            experiment_info = None

            # Iterate through the lines in the specified range to find the EXPERIMENT line
            for i in range(start_line, end_line):
                line = lines[i].strip()

                # Look for the line containing 'EXPERIMENT'
                if line.startswith('EXPERIMENT'):
                    # Extract the experiment description after the colon
                    experiment_info = line.split(':')[1].strip()

                if line.startswith('CROP') and 'CULTIVAR :' in line:
                    # Extract CULTIVAR information using split
                    parts = line.split('CULTIVAR :')
                    if len(parts) > 1:
                        cultivar = parts[1].split('ECOTYPE')[
                            0].strip()  # Extract everything between 'CULTIVAR :' and 'ECOTYPE'

                # Look for the line with simulation results (lines after @)
                if line.startswith('@'):

                    # Store the header line to return it
                    header_line = line

                    line_number = 0
                    for data_line in lines[i + 1:]:

                        line_number += 1

                        if not data_line.strip() or data_line.startswith('*'):
                            break

                        data_line = data_line.strip().split()
                        variable_name = ' '.join(data_line[:-2])  # Get the variable name
                        simulated_value = data_line[-2]
                        measured_value = data_line[-1]

                        # Replace any value starting with '-99' with an empty string
                        simulated_value = '' if simulated_value.startswith('-99') else simulated_value
                        measured_value = '' if measured_value.startswith('-99') else measured_value

                        # Append the row data
                        cultivar_data.append({
                            'TREATMENT': treatment,
                            'cultivar': cultivar,
                            'VARIABLE': variable_name,
                            'VALUE_SIMULATED': simulated_value,
                            'VALUE_MEASURED': measured_value,
                            'EXPERIMENT': experiment_info,
                            'POSITION': line_number,
                        })

                    # Convert to DataFrame and append to all_data
                    cultivar_df = pd.DataFrame(cultivar_data)
                    all_data = pd.concat([all_data, cultivar_df], ignore_index=True)

    # Remove rows where any of the columns 'VARIABLE', 'VALUE_SIMULATED', or 'VALUE_MEASURED' contain '--------'
    all_data = all_data[
        ~all_data[['VARIABLE', 'VALUE_SIMULATED', 'VALUE_MEASURED']].apply(lambda x: x.str.contains('--------')).any(
            axis=1)]

    # Convert the 'VALUE_SIMULATED' and 'VALUE_MEASURED' columns to numeric values
    all_data['VALUE_SIMULATED'] = pd.to_numeric(all_data['VALUE_SIMULATED'], errors='coerce')
    all_data['VALUE_MEASURED'] = pd.to_numeric(all_data['VALUE_MEASURED'], errors='coerce')

    # Split the 'Cultivar' column into 'treatment' and 'cultivar' columns
    # all_data[['treatment', 'cultivar']] = all_data['Cultivar'].str.split('_', expand=True)

    # Drop the original 'Cultivar' column as it's now split into two
    # all_data.drop(columns=['Cultivar'], inplace=True)

    # Convert all column names to lowercase
    all_data.columns = all_data.columns.str.lower()

    return all_data, header_line


def load_and_process_overview(calibration_code_list, base_data_dir,
                             treatments_dic, variable_name_mapping):
    """
        Loads, merges, and processes OVERVIEW.OUT and config.yaml files for each calibration code.

        Parameters
        ----------
        calibration_code_list : list of str
            List of subfolder names, e.g. ['cultivar_subset_a', ...]
        base_data_dir : str
            Path to the folder containing all calibration subfolders (e.g. '../data/ecotype_calibration')
        treatments_dic : dict
            Map short treatment codes to human-readable strings
        variable_name_mapping : dict
            Map raw DSSAT-like variable column names to human-readable ones
        extract_simulation_data : function
            Function to extract DataFrame from OVERVIEW.OUT (custom per-project)
        validate_file_path : function
            Function to check that a path is valid (custom per-project)

        Returns
        -------
        pd.DataFrame
            All processed, merged, and relabeled overview data
        """
    list_overview_df = []

    for calibration_code in calibration_code_list:
        print(calibration_code)

        # Full file path for this subset
        out_file_path = os.path.join(base_data_dir, calibration_code, 'OVERVIEW.OUT')
        # Validate that the file exists
        validated_path = validate_file_path(out_file_path)
        overview_df_raw, _ = extract_simulation_data(validated_path)
        # Remove rows where 'value_measured' column contains NaN values
        overview_df = overview_df_raw.dropna(subset=['value_measured']).copy()
        # Add calibration code as a new column
        overview_df['calibration_code'] = calibration_code

        # YAML meta extraction
        yaml_file_path = os.path.join(base_data_dir, calibration_code, 'config.yaml')
        with open(yaml_file_path) as config:
            try:
                config_file = yaml.safe_load(config)
            except yaml.YAMLError as exc:
                print(exc)

        for field in ['plantgro_variables', 'overview_variables', 'calibration_method', 'short_label', 'long_label']:
            value = config_file.get(field, [])
            overview_df[field] = ', '.join(value) if value else ''

        list_overview_df.append(overview_df)

    # Combine all into one big DataFrame
    overview_data = pd.concat(list_overview_df, ignore_index=True)

    # Unify and relabel columns
    overview_data.rename(columns={'treatment': 'treatment_cultivar'}, inplace=True)
    overview_data['treatment'] = overview_data['treatment_cultivar'].astype(str).str[:4]
    overview_data['treatment'] = overview_data['treatment'].map(treatments_dic)
    overview_data['variable'] = overview_data['variable'].map(variable_name_mapping)

    return overview_data


def update_variable_names(variable_names):
    """
    Format variable names with scientific notation for units
    """
    if isinstance(variable_names, str):
        variable_names = [variable_names]

    updated_names = []
    for var in variable_names:
        var = var.replace('g/m2', 'g m$^{-2}$')
        var = var.replace('kg/ha', 'kg ha$^{-1}$')
        var = var.replace('m2', 'm$^{-2}$')
        var = re.sub(r'\bm2\b', 'm$^{-2}$', var)  # Replace standalone "m2"
        updated_names.append(var)
    return updated_names
