# File to save images of beats in a specified directory

import os
import wfdb
import signal_api
import directory_structure
import natsort  # module used to sort file names
import wfdb
import pandas as pd


if __name__ == '__main__':

    # find directory where data is
    signal_dir = directory_structure.getReadDirectory('/content/irregular-heart-beats/mit-bih_waveform')

    # get all .hea and .dat files (respectively)
    signal_files = directory_structure.filesInDirectory('.hea', signal_dir)
    
    # sort file names in ascending order in list
    signal_files = natsort.natsorted(signal_files)

    # extract and save beats from file provided
    for signal_file in signal_files:
        print(signal_file)
        print (" directory structure :"+directory_structure.removeFileExtension(signal_file));
        signal_path = "/content/irregular-heart-beats/mit-bih_waveform/"+directory_structure.removeFileExtension(signal_file)
        #signal_dir + '/' + \
            
        print (" signal_path :"+signal_path);

        # get annotation data frame of signal file
        #ann = wfdb.rdann(signal_path, 'atr', return_label_elements=list(['symbol', 'description', 'label_store']), summarize_labels=True)
       
        # Specify the path to the signal file (excluding the extension)
        #signal_path = 'path_to_your_signal_file'

        # Read the annotation data using rdann
        ann = wfdb.rdann(signal_path, 'atr')

        # Create a DataFrame from the annotation data
        # 'sample' is the index of the sample in the signal file, 'symbol' refers to the annotation symbol
        annotation_df = pd.DataFrame({
            'sample': ann.sample,  # Sample index
            'symbol': ann.symbol,  # Annotation symbol (such as 'N', 'V', etc.)
            'description': ann.aux_note  # Optional description, if available
        })

        # Show the first few rows of the DataFrame
        print(annotation_df.head())

        # uncomment to save images of beats
        signal_api.extractBeatsFromPatient(signal_path, ann)