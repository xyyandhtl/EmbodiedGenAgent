from EG_agent.planning.btpg.behavior_tree.base_nodes.Action import Action
from EG_agent.planning.btpg.behavior_tree.base_nodes.Condition import Condition
from EG_agent.planning.btpg.behavior_tree.base_nodes.Inverter import Inverter
from EG_agent.planning.btpg.behavior_tree.base_nodes.Selector import Selector
from EG_agent.planning.btpg.behavior_tree.base_nodes.Sequence import Sequence


base_node_map = {
    "act": Action,
    "cond": Condition,
}

base_node_type_map={
    "act": "Action",
    "cond": "Condition",
}

composite_node_map = {
    "not": Inverter,
    "selector": Selector,
    "sequence": Sequence
}