
import os
import pandas as pd
import datetime
import re

import taxa
import audio
import report


def get_datafile_list(directory, file_number_limit):
    datafile_list = []

    try:
        objects = os.listdir(directory)
    except FileNotFoundError:
        print('Directory not found')
        return False
    else:
#        print(files)
        i = 0
        for filename in objects:
            if filename.lower().endswith(".results.csv"):
                datafile_list.append(filename)
                i = i + 1
            if i >= file_number_limit:
                break

        return datafile_list


# Todo: identify filename format: Audiomoth & SM4
def datetime_from_filename(filename):
    parts = filename.split(".")
    datepart = parts[0]

    # Note: order of these if-clauses are important.
    # Wildlife Acoustics SM4
    # TODO: Not UTC?
    pattern = re.compile("[A-Z0-9]{5}\_\d{8}\_\d{6}")
    if pattern.match(datepart):
        datepart = datepart[6:]
        return datetime.datetime.strptime(datepart, "%Y%m%d_%H%M%S")

    # Audiomoth firmaware 1.8.0, with T at the end
    pattern = re.compile("\d{8}\_\d{6}T")
    if pattern.match(datepart):
        datepart = datepart[0:14]
        return datetime.datetime.strptime(datepart, "%Y%m%d_%H%M%S")

    # Audiomoth firmaware 1.7.1
    pattern = re.compile("\d{8}\_\d{6}")
    if pattern.match(datepart):
        return datetime.datetime.strptime(datepart, "%Y%m%d_%H%M%S")

    # Audiomoth old firmware, 32-bit hexadecimal UNIX timestamp
    pattern = re.compile("[A-F0-9]{8}")
    if pattern.match(datepart):
        unix_seconds = int(datepart, 16)
        return datetime.datetime.fromtimestamp(unix_seconds)

    return datepart


def audio_filename_from_filename(filename, file_extension):
    # Have to have file extension as parameter provided by user, since it's not available on the Birdnet analysis files
    parts = filename.split(".")
    return parts[0] + "." + file_extension


def handle_files(dir, threshold):

    pd.io.formats.excel.ExcelFormatter.header_style = None

    file_extension = "wav" # Don't include dot here

    file_number_limit = 2000 # Limit for debugging
    filter_limit = float(threshold)

    max_segments_per_species = 5


    ###################################
    # Setup
    subdir_name = dir[(dir.rindex("/") + 1):]
    export_file_path = dir + "/_baim_" + subdir_name + ".xlsx"

    filtered_species_sheet_name = "Species conf " + str(filter_limit)

    datafile_list = get_datafile_list(dir, file_number_limit)

    # If Directory does not exist
    if not datafile_list:
        return False

    datafile_list.sort()

    print(datafile_list) # debug

    dataframe_list = []


    ###################################
    # Do batch operations for each file
    for filename in datafile_list:
        df = pd.read_csv(dir + "/" + filename)

        # Skip empty files
        if df.empty:
            continue

        # ~Audio filename
        df['Filename'] = audio_filename_from_filename(filename, file_extension)

        # Datetime
        df['File start'] = datetime_from_filename(filename)

        # Start time in h:m:s
        df['Start (h:m:s)'] = df.apply(lambda row: str(datetime.timedelta(seconds= row['Start (s)'])), axis = 1)

        dataframe_list.append(df)

        print("Handled file " + filename)

    # Combine per-file dataframes
    full_dataframe = pd.concat(dataframe_list, ignore_index=True)

    # Reorder columns
    new_index = ["Start (s)", "End (s)", "Common name", "Scientific name", "Filename", "File start", "Confidence", "Start (h:m:s)"]
    full_dataframe = full_dataframe[new_index]

    ###################################
    # Get list of species with high confidence
    # this dataframe is named just "df" for shortness and convention
    df = full_dataframe[full_dataframe['Confidence'] >= filter_limit]

    # Shuffle, so that audio snippets represent a random sample of the calls 
    df = df.sample(frac=1).reset_index(drop = True)
    #df.reset_index(inplace = True, drop = True)

    # Count species occurrences
    species_list = df.groupby(['Scientific name']).size()
    # This makes a dataframe with 0 as the column name
    species_dataframe = pd.DataFrame(species_list)
    # Rename column
    species_dataframe.columns = ['Count']
    # Sot descending
    species_dataframe.sort_values("Count", ascending = False, inplace = True)

    ###################################
    # Create Excel file
    writer = pd.ExcelWriter(export_file_path)

    full_dataframe.to_excel(writer, index=True, index_label="Row", sheet_name="Predictions", freeze_panes=(1, 1))
    species_dataframe.to_excel(writer, index=True, index_label="Row", sheet_name=filtered_species_sheet_name, freeze_panes=(1, 1))

    # Excel file settings
    #workbook  = writer.book
    worksheet_prediction = writer.sheets["Predictions"]
    worksheet_prediction.column_dimensions["D"].width = 20
    worksheet_prediction.column_dimensions["E"].width = 20
    worksheet_prediction.column_dimensions["F"].width = 20
    worksheet_prediction.column_dimensions["G"].width = 20

    worksheet_prediction.auto_filter.ref = worksheet_prediction.dimensions

    worksheet_species = writer.sheets[filtered_species_sheet_name]
    worksheet_species.column_dimensions["A"].width = 22

    writer.save()

    ###################################
    # Create report

    # Pick rows to make segments of

    picked_taxa = dict()
    picked_rows = dict()

    # First >=0.9
    for index in range(len(df)):
        sciname = df['Scientific name'].loc[index]

        # Skip if non-Finnish
        if taxa.is_non_finnish(sciname):
            continue

        # Skip if have enough
        if sciname in picked_taxa:
            if picked_taxa[sciname] >= max_segments_per_species:
                continue

        if (df['Confidence'].loc[index] >= 0.9):
            picked_rows[index] = sciname
            if sciname in picked_taxa:
                picked_taxa[sciname] = picked_taxa[sciname] + 1
            else:
                picked_taxa[sciname] = 1

    # Then 0.75-0.9
    for index in range(len(df)):
        sciname = df['Scientific name'].loc[index]

        # Skip if non-Finnish
        if taxa.is_non_finnish(sciname):
            continue

        # Skip if have enough
        if sciname in picked_taxa:
            if picked_taxa[sciname] >= max_segments_per_species:
                continue

        if (df['Confidence'].loc[index] >= filter_limit and df['Confidence'].loc[index] < 0.9):
            picked_rows[index] = sciname
            if sciname in picked_taxa:
                picked_taxa[sciname] = picked_taxa[sciname] + 1
            else:
                picked_taxa[sciname] = 1


    # TODO: check if the audio subdir name is data or Data

    # Sort by taxon name by recreating the dictionary
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    picked_rows = {k: v for k, v in sorted(picked_rows.items(), key=lambda item: item[1])}

    print(picked_rows)
    print(picked_taxa)

    segment_dir = dir + "/report"
    audio.create_dir(segment_dir)
    bird_report = report.report(segment_dir)


    print("========================")
    html = ""

    prev_taxon = ""

    for index, sciname in picked_rows.items():
        # Getting dataframe row as regular dict
        # "records" makes this to return dict without index, [0] takes the first and only row
        row = df.loc[[index]].to_dict("records")[0]
    #    print(row)

    #    print(row["Scientific name"])

        if row["Scientific name"] != prev_taxon:
            bird_report.add_taxon_divider(row["Scientific name"])
        prev_taxon = row["Scientific name"]


        audio_filepath = dir + "/Data/" + row['Filename']
        start_sec = int(row['Start (s)'])
        end_sec = int(row['End (s)'])

        props = dict(
            audio_filepath = audio_filepath,
            audio_filename = row['Filename'],
            segment_dir = segment_dir,
            start_sec = start_sec,
            end_sec = end_sec,
            scientific_name = row['Scientific name'],
            confidence = row['Confidence'],
            file_start_datetime = row['File start'],
            segment_start = row['Start (h:m:s)']
            )

        segment_filename = audio.make_audio_segment(props)

        bird_report.add_segment(props, segment_filename)

    return True

    #print(html)




    #print(species.non_finnish_species) # debug

# For debugging, running this file from command line
#handle_files("/mnt/c/Users/mikko/Documents/Audiomoth_2022/baimtest")