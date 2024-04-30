class Agent:

    def __init__(
        self,
        position: list,
        color: list,
        temporal_goal,
        selected_task,
        events,
    ):

        self.position = position
        self.color = color
        self.temporal_goal = temporal_goal
        self.state = temporal_goal.current_state - 1
        self.selected_task = selected_task
        self.events = events

    def get_position(self):
        return self.position

    def get_color(self):
        return self.color

    def get_state(self):
        return self.state

    def update_state(self, event):

        common_events = list(
            set(event) & set(self.events)
        )

        # print(common_events)
        # print(self.state)

        if not common_events == []:
            self.temporal_goal.step(common_events)
            self.state = self.temporal_goal.current_state - 1

    def reset_temporal_goal(self):
        self.temporal_goal.reset()
        self.state = self.temporal_goal.current_state - 1

    def get_temporal_goal(self):
        return self.temporal_goal

    def get_selected_task(self):
        return self.selected_task

    def set_selected_task(self, new_selected_task):
        self.selected_task = new_selected_task

    def get_events(self):
        return self.events
