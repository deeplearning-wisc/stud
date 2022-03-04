from .build import SSHEAD_REGISTRY, build_ss_head
# import all the ss head, so they will be registered
# from .cycle import CycleHead
# from .cycle_energy import CycleEnergyHead
# from .cycle_energy_1024_latter import CycleEnergy1024LatterHead
# from .cycle_energy_direct import CycleEnergyDirectHead
# from .cycle_energy_direct_add import CycleEnergyDirectAddHead
from .cycle_energy_direct_add_all import CycleEnergyDirectAddAllHead
# from .cycle_energy_direct_add_all_cache_new import CycleEnergyDirectAddAllCacheHead
# from .cycle_energy_direct_add_all_max import CycleEnergyDirectAddAllMaxHead
# from .cycle_energy_direct_add_all_mild_energy import CycleEnergyDirectAddAllMildHead
# from .cycle_energy_direct_add_all_noise import CycleEnergyDirectAddAllNoiseHead
# from .cycle_energy_direct_add_all_random import CycleEnergyDirectAddAllRandomHead
# from .cycle_energy_direct_add_att import CycleEnergyDirectAddAttHead
# from .cycle_energy_direct_add_att_neg import CycleEnergyDirectAddAttNegHead
# from .cycle_energy_direct_random import CycleEnergyDirectRandomHead
# from .cycle_energy_direct_max import CycleEnergyDirectMaxHead
# from .cycle_energy_direct_no import CycleEnergyDirectAddNoHead
from .rotation import RotationHead
from .jigsaw import JigsawHead

