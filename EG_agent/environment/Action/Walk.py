from EG_agent.environment.base.gen_action import GenAction

class Walk(GenAction):
    can_be_expanded = True
    num_args = 1

    def __init__(self, *args):
        super().__init__(*args)
        self.target_obj = self.args[0]

    @classmethod
    def get_info(cls, *arg):
        info = {}
        info["pre"]=set()
        info["add"]={f"IsNear(self,{arg[0]})"}
        info["del_set"] = {f'IsNear(self,{place})' for place in cls.valid_args if place != arg[0]}
        info["cost"] = 15
        return info

    def change_condition_set(self):
        # del_list = []
        # for c in self.agent.condition_set:
        #     if "IsNear" in c:
        #         del_list.append(c)
        # for c in del_list:
        #     self.agent.condition_set.remove(c)
        #
        # self.agent.condition_set.add(f"IsNear(self,{self.args[0]})")
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]
