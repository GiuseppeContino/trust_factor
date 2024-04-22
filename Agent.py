class Agent:

    def __init__(
        self,
        position: list,
        color: list,
        temporal_goal,
        events,
    ):

        self.position = position
        self.color = color
        self.temporal_goal = temporal_goal
        self.events = events

    def get_position(self):
        return self.position

    def get_color(self):
        return self.color

    def get_temporal_goal(self):
        return self.temporal_goal

    def get_events(self):
        return self.events
