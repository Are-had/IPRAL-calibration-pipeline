import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import curve_fit
import h5py
from scipy.ndimage import gaussian_filter1d 
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.dates as mdates
import pandas as pd
import glob
import os



def days_since_1970_to_datetime(days_array):
    epoch = datetime(1970, 1, 1)

    return [epoch + timedelta(days=float(d)) for d in days_array]  

def hours_since_1970_to_datetime(hours_array):
    epoch = datetime(1970, 1, 1)

    return [epoch + timedelta(hours=float(h)) for h in hours_array]  



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

def substract_bckgrd(rcs, bckgrd, r_square):
   

    data = ((rcs / r_square).T - bckgrd).T * r_square

    return np.array(data)


def read_file(path_file ) :
    
    """this function read a file and it returns a dictionnary """
    try :
        data = h5py.File(path_file , "r") 
        #print("File opened successfully!")
        return data
        
    except FileNotFoundError:
        print(f"Error: File not found at path {path_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        





def merged_signal(analog, photocounting, altitude , alltitude_fixed=14000,altitude_range=[13500,14500] , visual = False):

    mask = (altitude >= altitude_range[0]) & (altitude <= altitude_range[1] )
    K= np.nanmean(analog[:,mask],axis=1)/np.nanmean(photocounting[:,mask],axis=1)

    k_by_profile = K
    
    photocounting_k = np.zeros_like(photocounting, dtype=float)
    for i in range(len(k_by_profile)):
        photocounting_k[i,:] = photocounting[i,:] * k_by_profile[i]
        
        
    mask_up = altitude >= alltitude_fixed
    mask_down = altitude < alltitude_fixed


    selected_analog =analog[:,mask_down]
    selected_photocounting =photocounting_k[:,mask_up]
    merged_signal=np.concatenate(( selected_analog , selected_photocounting), axis=1)
    
    photocounting_k=np.nanmean(photocounting_k,axis=0)
    analog=np.nanmean(analog,axis=0)

    
    photocounting_k = gaussian_filter1d(photocounting_k, sigma=20)
    analog = gaussian_filter1d(analog, sigma=20)


    if (visual) : 
        plt.title("SIRTA analog and K photocountage")
        plt.plot(photocounting_k, altitude , label = "K * photocountage " )
        plt.plot(analog, altitude , label = "analog ")
        plt.legend()
        # plt.ylim((2000,40000))
    
    return merged_signal




def add_parallel_and_cross(rcs_p , bck_p , rcs_c , bck_c , alt ):
    alt_sqr =alt**2
    
    rcs_p_corrected = substract_bckgrd(rcs_p, bck_p, alt_sqr)
    rcs_c_corrected = substract_bckgrd(rcs_c, bck_c, alt_sqr)
    return rcs_c_corrected + rcs_p_corrected
    

def get_indx_from_range_time_sirta(file_name, start_time, end_time):

    data =read_file(file_name)
    sirta_data = data['SIRTA']
    
    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    
    t = np.array(sirta_data['time'])
    
    time = days_since_1970_to_datetime(t)
    indx_range = np.where((time >= start_time) & (time <= end_time))[0]
    
    return indx_range


def calculate_AMB_clear(file_name  ):
    

    data =read_file(file_name)

    conc = np.transpose(data["ATLID2_R100KM"]["conc"])
    atlid_altitude =data["ATLID2_R100KM"]["height"]

    conc=np.nanmean(conc, axis=0)
    atlid_altitude=np.nanmean(atlid_altitude, axis=0)
    conc=np.flip(conc)   # here we flip it to get the direction from down to up 
    atlid_altitude=np.flip(atlid_altitude)


    RAYLEIGH_CROSS_SECTION = 3.2897988e-31  # [m2 sr-1]
    beta_ray_355 = conc * RAYLEIGH_CROSS_SECTION
    alpha_ray_355 = beta_ray_355 / 0.119 
    n=conc.shape[0]
    AMB_clear = np.full(n, np.nan)

    dz = (np.diff(atlid_altitude))
    trapz_coef = ((alpha_ray_355[:-1] + alpha_ray_355[ 1:]) / 2) * dz
    sum1 = np.insert( np.nancumsum(trapz_coef) ,0,0 )

    AMB_clear = beta_ray_355 * np.exp(-2.0 * sum1)
    
    return AMB_clear , atlid_altitude # the resolution is the resolution of atlid altitude



def calibration_(merged_signal , AMB_clear , altitude_sirta , atlid_altitude, fit_mask1=[5000,18000] , fit_mask2=[18000,30000] , calibration_mask1=[8000 , 12000] , calibration_mask2=[18000 , 25000] ,N =60 , visual =True ):
    
    
    
    # -- fit the ATB_mol signal --
    
    def fit_func(x, gamma, delta, delta_rcs):
        return gamma * np.exp(delta * x) + delta_rcs

    x = altitude_sirta
    # y = np.nanmean(merged_signal,axis=0)
    y =merged_signal


    # Filter valid values
    mask1 = (x >= fit_mask1[0]) & (x <= fit_mask1[-1])
    x1 = x[mask1]
    y1 = y[mask1]
    popt1, _ = curve_fit(fit_func, x1, y1, p0=(1e6, -1e-4, 0), maxfev = 8000)
    y_model1 = fit_func(x1, *popt1)
    
    
    # Filter valid values
    mask2 = (x >= fit_mask2[0]) & (x <= fit_mask2[-1])
    x2 = x[mask2]
    y2 = y[mask2]
    popt2, _ = curve_fit(fit_func, x2, y2, p0=(1e6, -1e-4, 0), maxfev = 8000)
    y_model2 = fit_func(x2, *popt2)
    
    
    
    
    
    
    y_model=np.concatenate([y_model1,y_model2])
    x_fit= np.concatenate([x1,x2])

    # -- calculate the K and delta S of calibration -- 
    
    merged_signal_intr = np.interp(atlid_altitude, x_fit, y_model)   # atlid_altitude it is fliped so it is from down to up 

    
    
    np.random.seed(42)


    mask_8_12 = (atlid_altitude >= calibration_mask1[0]) & (atlid_altitude <= calibration_mask1[-1])
    indices_8_12 = np.where(mask_8_12)[0]
    selected_idx_8_12 = np.random.choice(indices_8_12, size=N, replace=True)

    # --- Selection in 20–25 km (20000–25000) ---
    mask_18_22 = (atlid_altitude >= calibration_mask2[0]) & (atlid_altitude <= calibration_mask2[-1])
    indices_18_22 = np.where(mask_18_22)[0]

    selected_idx_18_22 = np.random.choice(indices_18_22, size=N, replace=True)


    atlid_altitude_1= atlid_altitude[selected_idx_8_12]
    atlid_altitude_2 = atlid_altitude[selected_idx_18_22]

    S1= merged_signal_intr[selected_idx_8_12]
    S2=merged_signal_intr[selected_idx_18_22]

    AMB_1=AMB_clear[selected_idx_8_12]
    AMB_2=AMB_clear[selected_idx_18_22]

    K = (AMB_1-AMB_2)/(S1-S2)
    delta= (S2*AMB_1 - S1*AMB_2)/(AMB_1-AMB_2)

    
    K_avrg = np.mean(K)
    delta_avrg = np.mean(delta)




    if(visual) :

        # --- Plot both fits ---
        plt.figure(figsize=(8, 5))
        plt.plot(y, x, label="Original", alpha=0.4)
        plt.plot(y_model, x_fit, 'r--', label="Fit signal")
        plt.xlabel("RangeCorrectedSignal")
        plt.ylabel("Range (m)")
        plt.title("Piecewise Exponential Fit")
        # plt.ylim([5000, 30000])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
                
        
        
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




def calculate_SR(file_name ,ATB_sirta , Z_0 ) :
    
    
    
    data =read_file(file_name)

    conc = np.transpose(data["ATLID2_R100KM"]["conc"])
    atlid_altitude =data["ATLID2_R100KM"]["height"]
    altitude_sirta= np.array(data['SIRTA']["altitude"] )

    print(altitude_sirta)

    conc=np.nanmean(conc, axis=0)
    atlid_altitude=np.nanmean(atlid_altitude, axis=0)
    conc=np.flip(conc)   # here we flip it to get the direction from down to up 
    atlid_altitude=np.flip(atlid_altitude)
    RAYLEIGH_CROSS_SECTION = 3.2897988e-31  # [m2 sr-1]
    LR=50     # lidar ratio 

    
    ATB_sirta = np.interp(atlid_altitude, altitude_sirta, ATB_sirta)  #  this one no need because we alredy have ATB interpolled !
    
    
    
    beta_ray_355 = conc * RAYLEIGH_CROSS_SECTION
    alpha_ray_355 = beta_ray_355 / 0.119 
    
    SR= np.zeros(len(ATB_sirta))
    
    
    alpha_part = klett_inversion(atlid_altitude, ATB_sirta , alpha_ray_355 , Z_0)
    
    beta_part = alpha_part/LR

    
    SR=beta_part/beta_ray_355+1.0
    
    return SR 









def klett_inversion(Z, RCS, alpha , Z_0):

    N_Z = len(Z)
    dZ = np.diff(Z)  
    alpha_part = np.zeros(N_Z)
    
    
    good_index = np.where(abs(Z - Z_0) < 1000)[0]
    RCS_0 = np.nanmean(RCS[good_index])
    
    
    
    # Compute cumulative integral of RCS using trapezoidal rule

    int_sig = np.zeros(N_Z)
    int_sig[0] = 0.5 * RCS[0] * dZ[0]
    int_sig[1:] = np.nancumsum( (RCS[:-1] + RCS[1:])/2 * dZ)
    
    i_0 = np.argmin(np.abs(Z - Z_0))
    
    int_S = (int_sig-int_sig[i_0]) / RCS_0 


    #select the range of calculation     
    
    range_of_calculation =np.where(Z < Z_0)[0]
    
    alpha_part[range_of_calculation] = (RCS[range_of_calculation]/RCS_0) / (1.0/alpha[i_0] - 2.0 * int_S[range_of_calculation]) 
    
    
    alpha_max = -1e10
    for i_alt in range(1, i_0 + 1):
        if alpha_part[i_alt] > alpha_max:
            i_max = i_alt
            alpha_max = alpha_part[i_alt]
        if alpha_part[i_alt] < -alpha_max / 10 and alpha_max > 1e-2:
            alpha_part[i_alt:] = 0
            break
        
    alpha_part[alpha_part < 0] = 0
    
    return alpha_part 


