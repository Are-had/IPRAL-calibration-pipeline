from tools import * 





def L0_TO_L1(data , time_start , time_end , K_analog_par , K_analog_per , K_ph_par
            , K_ph_per , delta_analog_par , delta_analog_per , delta_ph_par ,
            delta_ph_per , mask_transition_parallel , mask_transition_perpendicular , AMB_clear ):


    # SIRTA data
    data_sirta = data['IPRAL']
    time_sirta= np.array(days_since_1970_to_datetime(data_sirta["time"]))
    alt_sirta = np.array(data_sirta["range"])
    index_time = get_indx_from_range_time_sirta(time_start , time_end , time_sirta)
    
    # Corrected signals
    rcs_02_rc = get_corrected_signal( data_sirta , alt_sirta ,rcs_="rcs_02")
    rcs_03_rc = get_corrected_signal( data_sirta , alt_sirta ,rcs_="rcs_03")
    rcs_04_rc = get_corrected_signal( data_sirta , alt_sirta ,rcs_="rcs_04")
    rcs_05_rc = get_corrected_signal( data_sirta , alt_sirta ,rcs_="rcs_05")



    # Average over the selected time interval
    analog_parallel=np.nanmean( rcs_02_rc[index_time , :] , axis=0 )
    analog_perpendicular=np.nanmean( rcs_04_rc[index_time , :] , axis=0 )
    photocounting_parallel=np.nanmean( rcs_03_rc[index_time , :] , axis=0 )
    photocounting_perpendicular=np.nanmean( rcs_05_rc[index_time , :] , axis=0 )
    
    
    # Calibrated ATB signals
    ATB_analog_par = K_analog_par * (analog_parallel - delta_analog_par)
    ATB_analog_per = K_analog_per * (analog_perpendicular - delta_analog_per)
    ATB_ph_par = K_ph_par * (photocounting_parallel - delta_ph_par)
    ATB_ph_per = K_ph_per * (photocounting_perpendicular - delta_ph_per)
    
    
    
    # merging signals
    ATB_par = merged_signal_hanning(ATB_analog_par, ATB_ph_par, alt_sirta, transition_start=mask_transition_parallel[0], transition_end=mask_transition_parallel[1])
    ATB_per = merged_signal_hanning(ATB_analog_per, ATB_ph_per, alt_sirta, transition_start=mask_transition_perpendicular[0], transition_end=mask_transition_perpendicular[1])

    # removing nans by interpolation
    ATB_per= remove_nans_interpolation(ATB_per, alt_sirta)
    ATB_par = remove_nans_interpolation(ATB_par, alt_sirta)
    
    
    # filtering 
    ATB_per = gaussian_filter(ATB_per , alt_sirta , max_sigma=5  )
    ATB_par = gaussian_filter(ATB_par , alt_sirta , max_sigma=5 )   
    ATB_total= ATB_per + ATB_par
    
    
    
    return ATB_total , ATB_par , ATB_per ,AMB_clear ,  alt_sirta






def L1_2_L2(ATB_par  , INDEX_FOR_THE_CALIBRATION , alt_sirta , beta_ray , LR =17 
            , reference_range = 50 , beta_aerosol_reference = 1e-9 ) :


    index_reference = np.argmin(np.abs(alt_sirta -INDEX_FOR_THE_CALIBRATION ))
    bin_length = np.abs(np.median(np.diff(alt_sirta)))

    print(f"calibration range {INDEX_FOR_THE_CALIBRATION -reference_range * bin_length } m  to {INDEX_FOR_THE_CALIBRATION + reference_range * bin_length } m")


    beta_aerosol, beta_sum = klett_backscatter_aerosol_simplifié( ATB_par,LR,beta_ray,index_reference,reference_range,beta_aerosol_reference,bin_length,8*np.pi/3 , affiche=False)

    return beta_aerosol , beta_sum