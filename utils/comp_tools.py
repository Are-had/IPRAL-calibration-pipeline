import numpy as np
from scipy.stats import pearsonr



def get_grouped_profiles(backscatter_atlid, alt_atlid, group_size):

    n_profiles = backscatter_atlid.shape[0]
    n_groups = n_profiles // group_size


    backscatter_atlid_grouped = []
    alt_atlid_grouped = []

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        
        # Moyenne des 50 profils
        mean_backscatter = np.nanmean(backscatter_atlid[start_idx:end_idx, :], axis=0)
        mean_alt = np.nanmean(alt_atlid[start_idx:end_idx, :], axis=0)
        
        backscatter_atlid_grouped.append(mean_backscatter)
        alt_atlid_grouped.append(mean_alt)

    backscatter_atlid_grouped = np.array(backscatter_atlid_grouped)
    alt_atlid_grouped = np.array(alt_atlid_grouped)

    return backscatter_atlid_grouped, alt_atlid_grouped






def interpolate_ipral_to_atlid(backscatter_ipral, alt_ipral, alt_atlid):

    # Ensure IPRAL altitudes are INCREASING 
    if alt_ipral[0] > alt_ipral[-1]:
        alt_ipral = alt_ipral[::-1]
        backscatter_ipral = backscatter_ipral[::-1]
    
    alt_atlid_increasing = alt_atlid[::-1]  # -422m → 40206m (INCREASING)
    
    # Interpolate IPRAL on the INCREASING ATLID grid
    backscatter_ipral_interp_increasing = np.interp(alt_atlid_increasing, alt_ipral,
        backscatter_ipral,
        left=np.nan,
        right=np.nan
    )

    # Reverse back to match original ATLID order (DECREASING)
    backscatter_ipral_interp = backscatter_ipral_interp_increasing[::-1]
    
    return backscatter_ipral_interp, alt_atlid





def calculate_correlation(backscatter_atlid, backscatter_ipral_interp, alt_atlid , mask):
    """
    Calculate Pearson correlation between ATLID and interpolated IPRAL backscatter
"""

    alt_min, alt_max = mask
    altitude_mask = (alt_atlid >= alt_min) & (alt_atlid <= alt_max)

    # Remove NaN and infinite values
    valid_mask = (
        np.isfinite(backscatter_atlid) & 
        np.isfinite(backscatter_ipral_interp) & 
        altitude_mask
    )
    
    backscatter_atlid_valid = backscatter_atlid[valid_mask]
    backscatter_ipral_valid = backscatter_ipral_interp[valid_mask]
    alt_valid = alt_atlid[valid_mask]
    
    if len(backscatter_atlid_valid) < 3:
        print("Not enough valid points for correlation")
        return None, None, 0
    
    # Calculate correlation
    correlation, p_value = pearsonr(backscatter_atlid_valid, backscatter_ipral_valid)
    
    return correlation, p_value