from btpg.behavior_tree.base_nodes import Action
from btpg.behavior_tree import Status

class GenAction(Action):
    can_be_expanded = True
    num_args = 1

    # 动态参数：不预先定义对象/位置集合，由上层逻辑/模型抽取。
    SurfacePlaces = set()
    SittablePlaces = set()
    CanOpenPlaces = set()
    CanPutInPlaces = set()
    Objects = set()
    HasSwitchObjects = set()

    AllObject = set()

    # ===================
    SURFACES = SurfacePlaces
    SITTABLE = SittablePlaces
    CAN_OPEN = CanOpenPlaces
    CONTAINERS = CanPutInPlaces
    GRABBABLE = Objects
    HAS_SWITCH = HasSwitchObjects

    @property
    def action_class_name(self):
        return self.__class__.__name__

    def change_condition_set(self):
        pass

    def update(self) -> Status:
        # 可选调用仿真脚本（如果存在）
        if self.num_args==1:
            script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1)']
        else:
            script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1) <{self.args[1].lower()}> (1)']
        self.env.run_script(script,verbose=True,camera_mode="PERSON_FROM_BACK") # FIRST_PERSON
        print("script: ",script)

        self.change_condition_set()
        return Status.RUNNING
