class BatchElement:
    def __init__(self, data):
        self.map = data['map']
        self.track_id = data['track_id']
        self.record_id = data['record_id']
        self.frame_id = data['frame_id']
        self.ego_positions = data['ego_positions']
        self.ego_headings = data['ego_headings']
        self.adjacent_positions = data['adjacent_positions']
        self.adjacent_headings = data['adjacent_headings']
        self.adjacent_visibility = data['adjacent_visibility']

        max_past_frames = 30


