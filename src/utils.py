import pandas as pd


def create_dataframe_chunk(recording_ids, track_ids, frames, visible_track_ids):
    """
    Creates a DataFrame chunk with columns: recordingId, trackId, frame, visibleTrackId.
    Each parameter is a list corresponding to multiple rows.
    """
    data = {
        'recordingId': recording_ids,
        'trackId': track_ids,
        'frame': frames,
        'visibleTrackId': visible_track_ids
    }
    return pd.DataFrame(data)
