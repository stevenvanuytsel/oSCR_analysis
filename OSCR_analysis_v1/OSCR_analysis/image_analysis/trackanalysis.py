import trackpy as tp
import pandas as pd

def filter_short_tracks(tracking_df, threshold):
    try:
        'frame' in tracking_df.columns and 'particle' in tracking_df.columns
    except KeyError:
        raise ValueError("Tracks must contain columns 'frame' and 'particle'.")

    grouped = tracking_df.reset_index(drop=True).groupby('particle')
    particles_in = len(tracking_df.particle.unique())
    short_in = grouped['frame'].count().min()

    filtered = grouped.filter(lambda x: x.frame.count() >= threshold)

    particles_out = len(filtered.particle.unique())
    short_out = filtered.groupby('particle')['frame'].count().min()
    print(f"Initially: {particles_in} tracks \t shortest track {short_in}\n"
            f"Filtered: {particles_out} tracks \t shortest track {short_out}")

    return filtered.reset_index(drop=True)

def individual_msd(tracking_df, microns_per_pixel, frames_per_second, **kwargs):
    imsd_df = tp.imsd(tracking_df, microns_per_pixel, frames_per_second, **kwargs)
    return imsd_df

def ensemble_msd(tracking_df, microns_per_pixel, frames_per_second, **kwargs):
    emsd_df = tp.emsd(tracking_df, microns_per_pixel, frames_per_second, **kwargs)
    return emsd_df
