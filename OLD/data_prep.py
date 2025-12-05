import h5py
from datetime import datetime
from netCDF4 import Dataset
import numpy as np
import os
import glob



def read_hdf5_file(path_file):
    """
    Reads an HDF5 file using h5py and returns an h5py.File object.
    """
    try:
        hdf5_obj = h5py.File(path_file, 'r')
        return hdf5_obj
    except FileNotFoundError:
        print(f"Error: File not found at path: {path_file}")
        return None
    except Exception as e:
        print(f"Error opening HDF5 file: {e}")
        return None




def read_nc_file(path_file) :
    
    #  """    This function reads a NetCDF (.nc) file and returns a netCDF4.Dataset object."""
    try:
        data = Dataset(path_file, "r")
        # File opened successfully!
        return data
    except FileNotFoundError:
        print(f"Error: File not found at path {path_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
def get_earlinet_files(directory_path):
    """
    Get all EARLINET .nc files from a directory
    
    """
    
    # Pattern to match EARLINET files
    pattern = os.path.join(directory_path, "EARLINET_*.nc")
    
    # Get all matching files
    files = glob.glob(pattern)
    
    # Sort files by name (this will sort by time)
    files.sort()
    
    print(f"Found {len(files)} EARLINET files:")
    for i, file in enumerate(files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    return files


        
def hdf5_to_datetime(hdf_data):
    """
    function to convert HDF5 time data to datetime array
    
    Parameters:
    -----------
    hdf_data : h5py.File object
        Opened HDF5 file with time data
        
    Returns:
    --------
    datetime_array : list of datetime objects
    """
    
    # Get time components
    years = hdf_data["year"][:]
    months = hdf_data["month"][:]
    days = hdf_data["day"][:]
    
    # Convert to datetime
    datetime_list = []
    for i in range(len(years)):
        if not (np.isnan(years[i]) or np.isnan(months[i]) or np.isnan(days[i])):
            dt = datetime(int(years[i]), int(months[i]), int(days[i]))
            datetime_list.append(dt)
        else : 
            datetime_list.append(np.nan)

            
    return datetime_list
   
   

def save_all_data_to_h5(hdf_data):
    """Save data with two groups: complete data and valid data only"""
    
    output_filename = "IPRAL_data_clean.h5"
    
    with h5py.File(output_filename, 'w') as f:
        
        # Convert time to continuous datetime
        time_datetime = hdf5_to_datetime(hdf_data)
        
        # Save time as datetime strings with NaN for invalid
        time_strings = []
        for dt in time_datetime:
            if not pd.isna(dt):
                time_strings.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                time_strings.append("NaN")
        
        # Find valid indices (where time is not "NaN")
        valid_indices = []
        for i, t in enumerate(time_strings):
            if t != "NaN":
                valid_indices.append(i)
        
        print(f"Total points: {len(time_strings)}")
        print(f"Valid points: {len(valid_indices)}")
        
        # GROUP 1: Complete data (all data including NaN)
        complete_group = f.create_group('complete_data')
        complete_group.create_dataset('time', data=time_strings)
        
        # GROUP 2: Valid data only (filtered)
        valid_group = f.create_group('valid_data')
        valid_time = [time_strings[i] for i in valid_indices]
        valid_group.create_dataset('time', data=valid_time)
        valid_group.create_dataset('indices', data=valid_indices)  # Store original indices
        
        # Save all measurement variables in both groups
        measurement_vars = ['altitude', 'pressure', 'temperature', 
                           'rcs_merged532', 'extinction_532', 'backscatter_532', 'scattering_ratio_532',
                           'rcs_merged355', 'extinction_355', 'backscatter_355', 'scattering_ratio_355']
        
        for var in measurement_vars:
            if var in hdf_data:
                # Complete data (all data)
                complete_data = hdf_data[var][:]
                complete_group.create_dataset(var, data=complete_data)
                
                # Valid data only (filtered by valid time indices)
                if var == 'altitude':
                    # Altitude is 1D, just copy it
                    valid_group.create_dataset(var, data=complete_data)
                else:
                    # Other variables are 2D (time, altitude) - filter by time
                    valid_data = complete_data[valid_indices]
                    valid_group.create_dataset(var, data=valid_data)
        
        # Save quality flags
        quality_vars = ['quality_flag_532', 'quality_flag_355']
        for var in quality_vars:
            if var in hdf_data:
                complete_data = hdf_data[var][:]
                complete_group.create_dataset(var, data=complete_data)
                
                # Filter for valid data
                valid_data = complete_data[valid_indices]
                valid_group.create_dataset(var, data=valid_data)
    
    print(f"✅ Data saved to: {output_filename}")
    print(f"✅ Groups created: 'complete_data' and 'valid_data'")
    return output_filename