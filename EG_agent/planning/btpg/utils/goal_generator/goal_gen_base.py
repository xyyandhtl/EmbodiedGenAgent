import random

class GoalGenerator:

    def __init__(self):
        # Mission-centric categories
        self.AREAS = {"corridor", "stairwell", "lobby", "room_a", "room_b", "exit", "charging_station"}
        self.TARGETS = {"victim", "hazard", "anomaly", "equipment", "leak"}
        # self.CAPTURE_MODES = {"photo", "video", "thermal"}  # optional; not enforced in goals

        # Predicate set aligned with actions: WALK -> IsNear, CAPTURE -> HasImage, MARK -> IsMarked, REPORT -> IsReported
        self.cond_pred = {"IsNear_", "IsEmergency_", "IsMarkable_", "IsAimed_"}

    def condition2goal(self, condition, easy=False):
        # easy kept for compatibility; not used in this simplified setup
        goal = ''
        if condition == 'IsNear_':
            area = random.choice(list(self.AREAS))
            goal = f'IsNear_(self,{area})'
        elif condition == 'IsEmergency':
            target = random.choice(list(self.TARGETS))
            goal = f'IsEmergency({target})'
        elif condition == 'IsMarkable':
            target = random.choice(list(self.TARGETS))
            goal = f'IsMarkable({target})'
        elif condition == 'IsAimed':
            target = random.choice(list(self.TARGETS))
            goal = f'IsReported({target})'
        return goal

    def get_goals_string(self, diffcult_type="multi"):
        goal_list = []
        if diffcult_type == "single":
            goal_mount = random.randint(1, 1)
        elif diffcult_type == "multi":
            goal_mount = random.randint(2, 3)
        elif diffcult_type == "mix":
            goal_mount = random.randint(1, 3)

        conditions = []
        for _ in range(goal_mount):
            condition = random.choice(list(self.cond_pred))
            conditions.append(condition)
        for condition in conditions:
            goal = self.condition2goal(condition)
            goal_list.append(goal)
        goal_string = ' & '.join(goal_list)
        return goal_string

    def random_generate_goals(self, n, diffcult_type="multi"):
        all_goals = []
        for _ in range(n):
            all_goals.append(self.get_goals_string(diffcult_type=diffcult_type))
        return all_goals