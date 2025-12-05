import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter1d 
from datetime import datetime, timedelta






def days_since_1970_to_datetime(days_array):
    epoch = datetime(1970, 1, 1)

    return [epoch + timedelta(days=float(d)) for d in days_array]  



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

# Function to save nested dict to HDF5
def save_dict_to_hdf5(h5file, path, dic):
    for key, item in dic.items():
        full_path = f"{path}/{key}"
        if isinstance(item, dict):
            save_dict_to_hdf5(h5file, full_path, item)
        elif isinstance(item, (np.ndarray, list)):
            h5file.create_dataset(full_path, data=np.array(item))
        elif isinstance(item, str):
            h5file.create_dataset(full_path, data=item.encode('utf-8'))
        elif isinstance(item, (int, float)):
            h5file.create_dataset(full_path, data=item)
        else:
            print(f"Warning: Could not save {full_path} of type {type(item)}")



def get_sigma(altitude, min_sigma=5, max_sigma=30, max_altitude=60000):
    # Linearly increase sigma with altitude
    slope = (max_sigma - min_sigma) / max_altitude
    return min_sigma + slope * altitude


def gaussian_filter(signal, altitude, window_size=500, min_sigma=2, max_sigma=20, max_altitude=10000):
    n_points = len(signal)
    filtered = np.zeros_like(signal)
    
    for i in range(n_points):
        half_window = window_size // 2
        start = max(0, i - half_window)
        end = min(n_points, i + half_window)
        window_signal = signal[start:end]
        
        sigma = get_sigma(altitude[i], min_sigma, max_sigma, max_altitude)
        
        filtered_window = gaussian_filter1d(window_signal, sigma=sigma)
        # Adjust index to get the value corresponding to the current point
        filtered[i] = filtered_window[i - start]
    
    return filtered




def get_corrected_signal(data_sirta  , alt   ,rcs_="rcs_12") :
  
    rcs = np.array(data_sirta[rcs_][:])
    back_rcs = np.array(data_sirta[f"bckgrd_{rcs_}"][:])
    rcs_rc = substract_bckgrd(rcs ,back_rcs , alt**2)
    
    return np.array(rcs_rc)



def substract_bckgrd(rcs, bckgrd, r_square):
   
    data = ((rcs / r_square).T - bckgrd).T * r_square

    return np.array(data)




def get_indx_from_range_time_sirta( start_time, end_time , time ):

    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    
    indx_range = np.where((time >= start_time) & (time <= end_time))[0]
    
    return indx_range



def plot_rcs(rcs , time , alt , title , vmax=None , vmin=0 , near_range = False , save=True) :
    
    plt.set_cmap("jet")
    plt.title(title)
    plt.pcolormesh(time , alt ,(rcs.T) , vmin = vmin , vmax=vmax )
    plt.colorbar(label="rcs ")
    
    if near_range :
        plt.ylim((0,15000))
    else :
        plt.ylim((0,40000))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Altitude (m)")
    if save :
        plt.savefig(f"figures/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()


def calculate_AMB_clear(conc , altitude_rs  ,altitude_ipral , depolarization_ratio=None ) :

    conc = np.interp(altitude_ipral, altitude_rs, conc)


    RAYLEIGH_CROSS_SECTION = 3.2897988e-31  # [m2 sr-1]
    beta_ray_355 = conc * RAYLEIGH_CROSS_SECTION
    alpha_ray_355 = beta_ray_355 / 0.119 
    n=conc.shape[0]
    
    AMB_clear = np.full(n, np.nan)

    dz = (np.diff(altitude_ipral))
    trapz_coef = ((alpha_ray_355[:-1] + alpha_ray_355[ 1:]) / 2) * dz
    sum1 = np.insert( np.nancumsum(trapz_coef) ,0,0 )

    AMB_clear = beta_ray_355 * np.exp(-2.0 * sum1)   
    
     
    return AMB_clear  , beta_ray_355 , alpha_ray_355




def conc_calculation (data_file_rs):

    Rd   = 287.              #gas constant for dry air [J/(kg.K)]
    Na = 6.022 * 10 **(23)
    Mair = 28.8*10**-3       # Masse molaire de l'air
    
    
    rs_data = read_nc_file(data_file_rs)
    
    alt= np.array(rs_data["alt"]) #alt max = 35408.438 m 

    P =np.array( rs_data["press"])    #air pressure hpa
    T=np.array(rs_data["temp"])   # temp_raw    K
    ro = (100 *P)/(Rd*T)
    
    conc= ro * (Na /Mair )
    
    
    return [conc , alt] 





def Calibration(rcs , AMB_clear , altitude , calibration_mask1=[2000 , 5000] , calibration_mask2=[6000 , 10000] ,N =60 , visual = True , seed=True) :
    
    if (seed) :
    
        np.random.seed(5)  
    

    mask_1 = (altitude >= calibration_mask1[0]) & (altitude <= calibration_mask1[-1])
    indices_1 = np.where(mask_1)[0]
    selected_idx_1 = np.random.choice(indices_1, size=N, replace=True)
    
    

    mask_2 = (altitude >= calibration_mask2[0]) & (altitude <= calibration_mask2[-1])
    indices_2 = np.where(mask_2)[0]
    selected_idx_2 = np.random.choice(indices_2, size=N, replace=True)

    S1= rcs[selected_idx_1]
    S2=rcs[selected_idx_2]

    AMB_1=AMB_clear[selected_idx_1]
    AMB_2=AMB_clear[selected_idx_2]

    K = (AMB_1-AMB_2)/(S1-S2)
    delta= (S2*AMB_1 - S1*AMB_2)/(AMB_1-AMB_2)


    K_avrg = np.nanmean(K)
    delta_avrg = np.nanmean(delta)




    if(visual) :

        
        # --- Plot histogram of K ---

        plt.figure(figsize=(8, 5))
        plt.hist(K, bins='auto', edgecolor='black', alpha=0.7 , label =f"the avrg of K is : {K_avrg}")
        plt.xlabel('K values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of K (n={N})')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        
        # --- Plot histogram of delta  ---
        
        plt.figure(figsize=(8, 5))
        plt.hist(delta, bins='auto', edgecolor='black', alpha=0.7 , label =f"the avrg of delta is :{delta_avrg}")
        plt.xlabel('delta values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of delta (n={N})')
        plt.grid(True)
        plt.legend()
        plt.show()





    return K_avrg , delta_avrg





def cost_function(params, rcs, amb_clear, correction_ranges, altitude):
    """
    correction_ranges: list of tuples [(min1, max1), (min2, max2), ...]
    """
    K_init, delta_init = params 
    
    # Créer un mask qui combine tous les ranges
    combined_mask = np.zeros_like(altitude, dtype=bool)
    
    for range_min, range_max in correction_ranges:
        mask = (altitude >= range_min) & (altitude <= range_max)
        combined_mask |= mask  # OR logique pour combiner les masks
    
    # Appliquer le mask combiné
    atb_masked = K_init * (rcs[combined_mask] - delta_init)
    amb_clear_masked = amb_clear[combined_mask]
    
    mape = np.nanmean(np.abs((atb_masked - amb_clear_masked) / amb_clear_masked)) * 100
    return mape


def optimize(rcs, AMB_clear, altitude, comparison_ranges, K_init, delta_init, method='Nelder-Mead'):
    """
    comparison_ranges: list of tuples for multiple ranges
    """
    from scipy.optimize import minimize
    
    initial_params = [K_init, delta_init]
    
    result = minimize(
        cost_function,
        initial_params, 
        args=(rcs, AMB_clear, comparison_ranges, altitude),
        method=method,
        options={'maxiter': 1000}
    )
    
    K_opt, delta_opt = result.x
    
    return K_opt, delta_opt, result









def klett_backscatter_aerosol_simplifié(S, l_aer, beta_mol, index_reference,reference_range, beta_aer_R0, bin_length, l_mol=8.73965404 , affiche = False):
    
    # calculat B : ----------------------------
    beta_mol_R0, S_R0 = get_reference_values(beta_mol, index_reference, S, reference_range)
    B = S_R0 / (beta_aer_R0 + beta_mol_R0)
    
    # calculat A : -------------------------------
    tau_integral_argument = (l_aer - l_mol) * beta_mol
    tau_integral = integrate_from_reference_trapezoid(tau_integral_argument, index_reference , bin_length)
    
    
    tau = np.exp(-2 * tau_integral)
    A = S * tau
    

    
    # calculat C : ------------------------------------
    C_integral_argument = l_aer * S * tau
    C_integral = integrate_from_reference_trapezoid(C_integral_argument, index_reference , bin_length)
    C = 2 * C_integral
    
    # Sum of aerosol and molecular backscatter coefficients.
    beta_sum = A / (B - C)
    
    
    # Aerosol backscatter coefficient.
    beta_aerosol = beta_sum - beta_mol
    
    if (affiche):
        relative_error = (beta_sum - beta_mol) / beta_mol * 100.0
        print('Mean relative error: ', np.mean(relative_error), '%')
        print('Max relative error: ', np.max(np.abs(relative_error)), '%')
        print('RMS relative error: ', np.sqrt(np.mean(relative_error**2)), '%')
    
    
    return beta_aerosol, beta_sum 




def get_reference_values(beta_molecular, index_reference, range_corrected_signal, reference_range):
    idx_min = index_reference - reference_range
    idx_max = index_reference + reference_range
    
    range_corrected_signal_reference = np.mean(range_corrected_signal[idx_min:idx_max+1])
    beta_molecular_reference = np.mean(beta_molecular[idx_min:idx_max+1]) 
    
    return beta_molecular_reference, range_corrected_signal_reference
    
    


def integrate_from_reference_trapezoid(integral_argument, index_reference, bin_length):
    """
    Calculate the cumulative integral of `integral_argument` from the reference point
    using trapezoidal integration.
    
    Parameters
    ----------
    integral_argument : array_like
        The argument to integrate (e.g., LR_part * RCS * exp(2*S_m))
    index_reference : integer
        The index of the reference height (bins)
    bin_length : float
        The vertical bin length (m)
    
    Returns
    -------
    integral : array_like
        The cumulative integral from the reference point
    """
    N_Z = len(integral_argument)
    integral = np.zeros(N_Z)
    
    # Set reference point to zero
    integral[index_reference] = 0.0
    
    # Below reference: integrate from ref down to beginning
    for i_Z in range(index_reference - 1, -1, -1):
        # Trapezoidal rule: (f[i+1] + f[i]) / 2 * dz
        integral[i_Z] = integral[i_Z + 1] - 0.5 * (integral_argument[i_Z + 1] + 
                                                     integral_argument[i_Z]) * bin_length
    
    # Above reference: integrate from ref up to end
    for i_Z in range(index_reference + 1, N_Z):
        # Trapezoidal rule: note the sign change
        integral[i_Z] = integral[i_Z - 1] + 0.5 * (integral_argument[i_Z - 1] + 
                                                     integral_argument[i_Z]) * bin_length
    
    return integral














def merged_signal_hanning(analog, photocounting, altitude, transition_start=15000, transition_end=20000):

    merged_signal = np.zeros_like(analog)
    
    mask_low = altitude < transition_start
    mask_high = altitude > transition_end
    mask_transition = (altitude >= transition_start) & (altitude <= transition_end)
    
    merged_signal[mask_low] = analog[mask_low]
    merged_signal[mask_high] = photocounting[mask_high]
    
    # Zone de transition avec fenêtre de Hanning
    if np.any(mask_transition):
        n_trans = np.sum(mask_transition)

        hanning_window = 0.5 * (1 - np.cos(np.pi * np.arange(n_trans) / (n_trans - 1)))
        
        merged_signal[mask_transition] = (
            (1 - hanning_window) * analog[mask_transition] + 
            hanning_window * photocounting[mask_transition]
        )
    
    # merged_signal_filtered = gaussian_filter1d(merged_signal, sigma=6)
    
    return merged_signal






def remove_nans_interpolation(signal , altitude, N_valid=10):

    signal_clean = np.nan_to_num(signal, nan=0.0)
    mask_valid = ~np.isnan(signal)
    if np.sum(mask_valid) > N_valid:  # Au moins N_valid points valides
        signal_clean = np.interp(altitude, altitude[mask_valid], signal[mask_valid])
    else:
        signal_clean = np.nan_to_num(signal, nan=1e-7)
    return signal_clean







# def correct_atb(ATB_SIRTA , AMB_clear):
    
#     bias = ATB_SIRTA - AMB_clear 
#     ATB_SIRTA_corrected = ATB_SIRTA
#     ATB_SIRTA_corrected[bias<0] = AMB_clear[bias<0]
#     return ATB_SIRTA_corrected