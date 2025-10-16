import yaml, pprint, math
from scipy.special import ellipk
from scipy.constants import *
import numpy as np

from phidl import quickplot as qp
from phidl import Device
import phidl.geometry as pg
import phidl.path as pp

STRING_TO_OBJECT = {
    "pp.arc": pp.arc,
}

def check_config_key(config, string):

    """
    Check if keys in the config dictionary contains specific strings.
    """
    
    for key in config.keys():
        if string in key:
            return True
    
    return False

def replace_special_strings(obj):

    """辞書やリストを再帰的に探索し、特殊な文字列を対応するオブジェクトに変換"""
    if isinstance(obj, dict):
        return {k: replace_special_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_special_strings(v) for v in obj]
    elif isinstance(obj, str) and obj in STRING_TO_OBJECT:
        return STRING_TO_OBJECT[obj]
    else:
        return obj

def set_JJtype(config, filename):
    
    filename = str(filename)
    if not "JJ" in filename:
        print(filename)
        pass
    else:
        # print("""
        # Some configuration values are set depending on the loaded file name.
        # JJ_type : manhattan or dolan
        # JJ_photolitho : True or False
        # """)

        if "photolitho" in filename:
            config["JJ_photolitho"] = True
        else:
            config["JJ_photolitho"] = False

        if "manhattan" in filename:
            config["JJ_type"] = "manhattan"
        elif "dolan" in filename:
            config["JJ_type"] = "dolan"
        else:
            raise ValueError("Correct JJ type not specified!!")

    return config

# YAML 設定ファイルを読み込む関数
def load_config(file_path):

    def _load_config( _file_name ):
        with open(_file_name, 'r') as _file:
            _config = yaml.safe_load(_file)
        _flat_data = flatten_dict(_config)
        _global_data = {f"{k}": v for k, v in _flat_data.items()}
        _global_data = replace_special_strings(_global_data)
        return _global_data
    
    config = {}
    if type(file_path) == list: # For list type, config will be overwritten by value in the right 
        for file in file_path:
            config = {**config, **_load_config( file )}
            config = set_JJtype(config, file)
    else:
        config = _load_config( file_path )
        config = set_JJtype(config, file_path)
    
    return config

# 再帰的にフラットな変数名で辞書を展開
def flatten_dict(d, parent_key="", sep="_"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# def flatten_dict(d, parent_key="", sep="_"):
#     items = []
#     for k, v in d.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         elif isinstance(v, list):
#             # list の中身が dict なら flatten
#             new_list = []
#             for elem in v:
#                 if isinstance(elem, dict):
#                     new_list.append(flatten_dict(elem, sep=sep))
#                 else:
#                     new_list.append(elem)
#             items.append((new_key, new_list))
#         else:
#             items.append((new_key, v))
#     return dict(items)

def phidl_to_metal(device_list, outname):

    chipdesign_qiskit = Device('chipdesign_qiskit')
    chipdesign_qiskit_pocket = Device('chipdesign_qiskit_pocket')

    for ilayer, device in enumerate(device_list):
        pocket = device["device"].pocket
        metal  = device["device"].metal
        for i in pocket.get_layers():
            chipdesign_qiskit_pocket.add_ref( pg.copy_layer(pocket, layer = i, new_layer=ilayer) )
        for i in metal.get_layers():
            chipdesign_qiskit.add_ref( pg.copy_layer(metal, layer = i, new_layer=ilayer) )            

    chipdesign_qiskit = pg.union( chipdesign_qiskit, by_layer = True )
    chipdesign_qiskit_pocket = pg.union( chipdesign_qiskit_pocket, by_layer = True )
    chipdesign_qiskit.flatten()
    chipdesign_qiskit_pocket.flatten()
    qp(chipdesign_qiskit)
    qp(chipdesign_qiskit_pocket)
    chipdesign_qiskit.write_gds(f'output/qiskit-metal/{outname}.gds')
    chipdesign_qiskit_pocket.write_gds(f'output/qiskit-metal/{outname}_pocket.gds')


    # Dump port data
    data = {}
    for ilayer, device in enumerate(device_list):
        key =  device["name"]
        data[key] = dict(
            layer = ilayer
        )

        port_data = {}
        jj_data = {}        
        for port in device["device"].pocket.get_ports():
            if "LaunchPad" in str(port.name):
                name, gap = port.name.split('_')
                start, end = phidl_port_to_metal_pin(port)
                port_data[name] = dict(
                    start = start,
                    end   = end,
                    width = float(port.width),
                    gap   = float(gap),
                )
            elif port.name == "Junction_up":
                jj_data["start"] = [float(port.midpoint[0]), float(port.midpoint[1])]
            elif port.name == "Junction_down":
                jj_data["end"] = [float(port.midpoint[0]), float(port.midpoint[1])]           
                jj_data["width"] = float(port.width)

        if port_data:
            data[key]["ports"] = port_data
        if jj_data:
            data[key]["jj"] = jj_data            

    pprint.pprint(data)
    with open(f'output/qiskit-metal/{outname}.yaml', 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def rename_port(device, before, after):

    device.ports[after] = device.ports[before]
    device.ports[after].name = after
    del device.ports[before]

def extract_with_ports(device, layers_to_extract):

    # print(device.get_ports())
    extracted = pg.extract(device, layers_to_extract)
    
    for port in device.get_ports():
        if port.name not in [x.name for x in extracted.get_ports()]:
            extracted.add_port(
                name=port.name, 
                midpoint=port.midpoint, 
                width=port.width, 
                orientation =port.orientation
            )
    
    return extracted

def boolean_with_ports(deviceA, deviceB, logic, layer):

    boolean = pg.boolean(deviceA, deviceB, logic, layer = layer)
    
    for port in deviceA.get_ports() + deviceB.get_ports():
        if port.name not in [x.name for x in boolean.get_ports()]:
            boolean.add_port(
                name=port.name, 
                midpoint=port.midpoint, 
                width=port.width, 
                orientation =port.orientation
            ) 

    return boolean

def phidl_port_to_metal_pin(port):
    x0, y0 = port.midpoint
    theta_rad = np.deg2rad(port.orientation + 90)  # orientationに90度足す（垂直方向）

    dx = (port.width / 2) * np.cos(theta_rad)
    dy = (port.width / 2) * np.sin(theta_rad)

    point1 = [float(x0 + dx), float(y0 + dy)]
    point2 = [float(x0 - dx), float(y0 - dy)]

    return point1, point2

def get_relative_permittivity(material):

    if material == "silicon":
        eps_r = 11.9
        #eps_r = 11.45
    elif material == "sapphire":
        eps_r = 9.4
    else:
        raise ValueError(f"Unsupported material: {material}")
    
    return eps_r

def calculate_effective_permittivity(core_width, gap_width, height, material):

    w = core_width * 1e-6
    s = gap_width * 1e-6
    h = height * 1e-6

    eps_r = get_relative_permittivity(material)

    k0 = w/(w + 2*s)
    k0_prime = math.sqrt(1-pow(k0, 2))
    k3 = math.tanh((math.pi*w)/(4*h))/math.tanh((math.pi*(w+2*s))/(4*h))
    k3_prime = math.sqrt(1-pow(k3, 2))
    K_k0 = ellipk(k0**2)
    K_k0_prime=ellipk(k0_prime**2)
    K_k3 = ellipk(k3**2)
    K_k3_prime=ellipk(k3_prime**2)
    K_tilde = (K_k0_prime/K_k0)*(K_k3/K_k3_prime)
    eps_eff = (1 + eps_r * K_tilde)/(1 + K_tilde)
    return eps_eff

def calculate_effective_velocity(core_width, gap_width, height, material):
    """有効伝搬速度を返す共通処理"""

    eps_eff = calculate_effective_permittivity(core_width, gap_width, height, material)

    return c / math.sqrt(eps_eff)

def calculate_resonator_frequency(
        length = 3000, # um
        core_width = 10, # um
        gap_width = 6, # um
        height = 525, # um
        material = "silicon"
        ):

    # convert um to m
    l = length * 1e-6
    c_eff = calculate_effective_velocity(core_width, gap_width, height, material)
    f = c_eff / (4*l)

    return f*1e-6

def calculate_resonator_length(
        frequency = 5000, # MHz
        core_width = 10, # um
        gap_width = 6, # um
        height = 525, # um
        material = "silicon"
        ):

    # convert um to m & MHz to Hz
    f = frequency * 1e+6
    c_eff = calculate_effective_velocity(core_width, gap_width, height, material)
    l = c_eff / (4*f)

    return l*1e+6

def calculate_purcellfilter_frequency(
        edge1,
        edge2,
        length = 3000, # um
        core_width = 10, # um
        gap_width = 6, # um
        height = 525, # um
        material = "silicon"
        ):

    # convert um to m
    c_eff = calculate_effective_velocity(core_width, gap_width, height, material)

    # effective length
    # eps_eff = calculate_effective_permittivity(core_width, gap_width, height, material)
    # edge1_cap = eps_eff * edge1["finger_length"] * edge1["finger_width"] / edge1["cap_gap"] * edge1["n_step"]
    # eff_len_edge1 = edge1_cap * 50 / math.pi

    # edge2_cap = eps_eff * edge2["finger_length"] * edge2["finger_width"] / edge2["cap_gap"] * edge2["n_step"]
    # eff_len_edge2 = edge2_cap * 50 / math.pi

    eff_len_edge1 = (edge1["finger_length"] + edge1["finger_width"]) * edge1["n_step"]
    eff_len_edge2 = (edge2["finger_length"] + edge2["finger_width"]) * edge2["n_step"]

    print("eff_len_edge1", eff_len_edge1)
    print("eff_len_edge2", eff_len_edge2)

    length = length + eff_len_edge1 + eff_len_edge2

    l = length * 1e-6
    f = c_eff / (2*l)

    return f*1e-6