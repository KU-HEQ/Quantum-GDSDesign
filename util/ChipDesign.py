import sys, copy
from qubit_templates import *
from functions import *

def resolve_from_string(path: str, scope=None):
    if scope is None:
        scope = locals()

    obj_name, *attrs = path.split(".")
    obj = scope[obj_name] 
    for a in attrs:
        obj = getattr(obj, a)
    return obj

def deep_set(config, path, value, sep=":"):
    keys = path.split(sep)
    *parents, last = keys
    cur = config
    for k in parents:
        if k.isdecimal():
            k = int(k)
        cur = cur[k]
    if last.isdecimal():
        last = int(last)
    cur[last] = value

def init_chipdesign(config, param_x, param_y, x, y):

    # param_x
    if (param_x is None):
        pass
    elif type(param_x) is list:
        for par, var in zip(param_x, x):
            if ":" in par:
                deep_set(config, par, var)
            else:
                config[par] = var
    elif ":" in param_x:
        deep_set(config, param_x, x)
    elif ((param_x in config) or (param_x == "dummy")):
        config[param_x] = x
    else:
        raise ValueError(f"{param_x} is not in config dictionary!!")
    
    # param_y    
    if (param_y is None):
        pass    
    elif type(param_y) is list:
        for par, var in zip(param_y, y):
            if ":" in par:
                deep_set(config, par, var)
            else:
                config[par] = var
    elif ":" in param_y:        
        deep_set(config, param_y, y)
    elif ((param_y in config) or (param_y == "dummy")):
        config[param_y] = y
    else:
        raise ValueError(f"{param_y} is not in config dictionary!!")


def sweep_chipdesign( config, userfunction = None ):

    def _sweep_chipdesign( config, userfunction = None, sweep = True, first_call = True, param_x = None, param_y = None, x = None, y = None, n_level = None ):

        if config["Grid_sweep_type"] == "array":
            devices = []
            for device in config["Grid_sweep_devices"]:
                updated_config = copy.deepcopy(config) # This must be deepcopy !! (Otherwise original config will be overwrited...)
                for key in device:
                    init_chipdesign(updated_config, key, None, device[key], None)
                devices.append( globals()["chipdesign_" + config["Grid_name"]](updated_config, sweep = sweep) )
            devices.append( globals()["chipdesign_" + config["Grid_name"]](updated_config, sweep = sweep, only_frame = True) )
            device_list = []
            for row in config["Grid_sweep_array"]:
                for cell in row:
                    if cell is None:
                        device_list.append( devices[-1] )
                    else:
                        device_list.append( devices[cell] )
            D = pg.grid(
                device_list,
                spacing = (config["Grid_sweep_gap_x"] * config["Frame_size_width"], config["Grid_sweep_gap_y"] * config["Frame_size_height"]),
                shape = np.array(config["Grid_sweep_array"], dtype=object).T.shape
            )
        elif config["Grid_sweep_type"] == "gridsweep":
            if first_call:
                n_level = len(config["Grid_sweep_array"]) - 1
            else:
                init_chipdesign(config, param_x, param_y, x, y)
                # count_down
                n_level -= 1
            array = config["Grid_sweep_array"][n_level]

            param_defaults = {}
            if n_level == 0:
                if userfunction:
                    function = userfunction
                else:
                    function = globals()["chipdesign_" + config["Grid_name"]]
            else:
                function = _sweep_chipdesign
                param_defaults = { 
                    'n_level' : n_level,
                    'first_call' : False,
                }
            param_defaults = {
                **param_defaults, 
                'sweep' : True,
                'param_x' : array["param_x"],
                'param_y' : array["param_y"],   
                'config' : config,
            }

            D = pg.gridsweep(
                function = function,
                param_x = {'x' : array["x"]},
                param_y = {'y' : array["y"]},
                param_defaults = param_defaults,
                spacing = (
                    array["gap_x"] * config["Frame_size_width"], 
                    array["gap_y"] * config["Frame_size_height"],
                ),
                label_layer = None
            )
            D.center = (0, 0)
        else:
            raise ValueError("Incorrect Grid_sweep_type !!")

        return D
    
    return _sweep_chipdesign( config, userfunction )

def chipdesign_wrapper(func):
    def wrapper(config, sweep=False, param_x=None, param_y=None, x=None, y=None, only_frame=False):
        init_chipdesign(config, param_x, param_y, x, y)
        chipdesign = Device('chipdesign')
        devicelist = []
        FM = device_Frame(config)
        if only_frame or ((param_x is not None and x is None) or (param_y is not None and y is None)):
            chipdesign.add_ref(FM)
            return chipdesign

        # chipdesign, devicelist = func(config, chipdesign, devicelist)
        func(config, chipdesign, devicelist)
        chipdesign.add_ref(FM)

        if sweep:
            return chipdesign
        else:
            return chipdesign, devicelist
    return wrapper

def chipdesign_Test(config, param_x, param_y, x, y):

    init_chipdesign(config, param_x, param_y, x, y)

    chipdesign = Device('chipdesign')

    # Frame
    FM=Device('frame')
    rectangle = pg.rectangle((config["Frame_size_width"], config["Frame_size_height"]), config["Frame_layer"])
    FM.add_ref( pg.invert(rectangle, border = config["Frame_width"], precision = 1e-6, layer = config["Frame_layer"]) )
    FM.center = (0, 0)
    chipdesign.add_ref(FM)

    return chipdesign


# def chipdesign_transmon2D(config, param_x = None, param_y = None, x = None, y = None, only_frame = False):
@chipdesign_wrapper
def chipdesign_transmon2D(config, chipdesign, devicelist):

    # init_chipdesign(config, param_x, param_y, x, y)

    # chipdesign = Device('chipdesign')

    # FM = device_Frame(config)
    
    # if only_frame or ((param_x is not None and x is None) or (param_y is not None and y is None)):
    #     chipdesign.add_ref(FM)
    #     return chipdesign

    FL = device_FeedLine(config)
    chipdesign.add_ref(FL.device)

    CP = device_CornerPoints(config)
    chipdesign.add_ref(CP)

    TA = device_TestAreas(config, DCLine = config["DCLine_activate"])
    chipdesign.add_ref(TA)

    R = []
    for i, resonator_config in enumerate(config["Resonator_devices"]):

        # Set entangle flag
        if config.get("JJ_entangle") and any(i in entangle_config["pairs"] for entangle_config in config["JJ_entangle"]):
            resonator_config.update( entangle = True )

        R.append( device_Resonator(config, **resonator_config) )
        if "movex" in resonator_config:
            R[i].movex( resonator_config["movex"] )
        if "movey" in resonator_config:
            R[i].movey( resonator_config["movey"] )
        if "xmax" in resonator_config:
            xmax_ref = resolve_from_string(path = resonator_config["xmax"], scope=locals()) if type(resonator_config["xmax"]) == str else resonator_config["xmax"]
            R[i].xmax = xmax_ref - resonator_config["feedline_gap"]
        if "ymax" in resonator_config:
            ymax_ref = resolve_from_string(path = resonator_config["ymax"], scope=locals()) if type(resonator_config["ymax"]) == str else resonator_config["ymax"]
            R[i].ymax = ymax_ref - resonator_config["feedline_gap"]

        chipdesign.add_ref(R[i].device)
    

    if config["JJ_entangle"]:
        line = []
        for i, entangle_config in enumerate(config["JJ_entangle"]):
            pairs = entangle_config["pairs"]
            port_config = dict(
                port1 = R[ pairs[0] ].device.references[1].ports['entangle'],
                port2 = R[ pairs[1] ].device.references[1].ports['entangle']
            )
            line.append( device_EntangleLine({**port_config, **entangle_config["path"]}) )
            chipdesign = pg.boolean(chipdesign, line[i].metal, 'not', layer = 4)
            chipdesign.add_ref( line[i].device )

    if config["DCLine_activate"]:
        DC = device_DCLine(config) 
        chipdesign = pg.boolean(chipdesign, DC.metal, 'not', layer = 4)
        chipdesign.add_ref(DC.device)


    BX = device_TestBoxes(DCLine = config["DCLine_activate"])
    chipdesign.add_ref(BX)

    JJ = []
    for i, jj_config in enumerate(config["JJ_devices"]):
        JJ.append( device_JJ({**config, **jj_config}) )         

        if "movex" in jj_config:
            JJ[i].movex( jj_config["movex"] )
        if "movey" in jj_config:
            JJ[i].movey( jj_config["movey"] )

        chipdesign.add_ref(JJ[i])

    EB = []
    for i, eb_config in enumerate(config["EBLine_devices"]):
        EB.append( device_EBLine(config) )       

        if "movex" in eb_config:
            EB[i].movex( eb_config["movex"] )
        if "movey" in eb_config:
            EB[i].movey( eb_config["movey"] )   

        chipdesign.add_ref(EB[i])

    # chipdesign.add_ref(FM)

    devicelist = [
        dict(device = FL, name = "FeedLine"),
        dict(device = R[0], name = "Qubit1"),    
        dict(device = R[1], name = "Qubit2"),
        dict(device = R[2], name = "Qubit3"),    
        dict(device = R[3], name = "Qubit4")
    ]

    return chipdesign, devicelist

# def chipdesign_transmon2D_Purcell(config, param_x = None, param_y = None, x = None, y = None, only_frame = False):
@chipdesign_wrapper
def chipdesign_transmon2D_Purcell(config, chipdesign, devicelist):
    # init_chipdesign(config, param_x, param_y, x, y)

    # chipdesign = Device('chipdesign')

    # FM = device_Frame(config)
    
    # if only_frame or ((param_x is not None and x is None) or (param_y is not None and y is None)):
    #     chipdesign.add_ref(FM)
    #     return chipdesign

    FL = device_FeedLine_PurcellFilter(config)
    chipdesign.add_ref(FL.device)

    R = []
    for i, resonator_config in enumerate(config["Resonator_devices"]):

        # Set entangle flag
        if config.get("JJ_entangle") and any(i in entangle_config["pairs"] for entangle_config in config["JJ_entangle"]):
            resonator_config.update( entangle = True )

        R.append( device_Resonator(config, **resonator_config) )
        if "movex" in resonator_config:
            R[i].movex( resonator_config["movex"] )
        if "movey" in resonator_config:
            R[i].movey( resonator_config["movey"] )
        if "xmax" in resonator_config:
            xmax_ref = resolve_from_string(path = resonator_config["xmax"], scope=locals()) if type(resonator_config["xmax"]) == str else resonator_config["xmax"]
            R[i].xmax = xmax_ref - resonator_config["feedline_gap"]
        if "ymax" in resonator_config:
            ymax_ref = resolve_from_string(path = resonator_config["ymax"], scope=locals()) if type(resonator_config["ymax"]) == str else resonator_config["ymax"]
            R[i].ymax = ymax_ref - resonator_config["feedline_gap"]
        if "ymin" in resonator_config:
            ymin_ref = resolve_from_string(path = resonator_config["ymin"], scope=locals()) if type(resonator_config["ymin"]) == str else resonator_config["ymin"]
            R[i].ymin = ymin_ref + resonator_config["feedline_gap"]

        chipdesign.add_ref(R[i].device)

    # chipdesign.add_ref(FM)

    devicelist = [
        dict(device = FL, name = "FeedLine"),
        dict(device = R[0], name = "Qubit1"),    
        dict(device = R[1], name = "Qubit2"),
        dict(device = R[2], name = "Qubit3"),    
        dict(device = R[3], name = "Qubit4")
    ]

    #return chipdesign, devicelist

@chipdesign_wrapper
def chipdesign_transmon3D(config, chipdesign, devicelist):

    # init_chipdesign(config, param_x, param_y, x, y)

    # chipdesign = Device('chipdesign')

    # FM = device_Frame(config)
    
    # if only_frame or ((param_x is not None and x is None) or (param_y is not None and y is None)):
    #     chipdesign.add_ref(FM)
    #     return chipdesign

    PAD=Device('PAD')
    rectangle = pg.rectangle(( config["Pad_width"], config["Pad_height"]), config["Pad_layer"])
    rectangle.polygons[0].fillet( config["Pad_rounding"] )
    PAD.add_ref( rectangle ).movex(0).movey(0.5*config["Pad_gap"])
    PAD.add_ref( rectangle ).mirror(p1 = (0, 0), p2 = (200, 0)).movex(0).movey(-0.5*config["Pad_gap"])
    PAD.center = (0, 0)

    chipdesign.add_ref(PAD)

    JJ       = device_JJ({**config, **dict(JJ_squid = False)})          
    JJ_squid = device_JJ({**config, **dict(JJ_squid = True)})  
    if config["JJ_squid"]:
        chipdesign.add_ref(JJ_squid)
    else:
        chipdesign.add_ref(JJ)

    chipdesign = pg.union( chipdesign, layer = config["Pad_layer"] )
    for pol in chipdesign.polygons: # unions are separated in dolan structure, so loop through all polygons
        pol.fillet( config["Pad_JJ_rounding"] )
    chipdesign = pg.union( chipdesign, layer = config["Pad_layer"] )

    text = eval(config["Text_string"], {"width": x, "height": y})
    move_x = config["Text_pos_x"]*0.5*config["Frame_size_width"]
    move_y = config["Text_pos_y"]*0.5*config["Frame_size_height"]    

    T = pg.text(text, size=config["Text_size"], layer = config["Text_layer"])
    T.center=(0,0)
    T.move([move_x,move_y])
    chipdesign.add_ref(T)

    TA = Device('TestArea')
    rectangle = pg.rectangle(( config["TestPoint_box_width"], config["TestPoint_box_length"]), config["TestPoint_layer"])
    rectangle.polygons[0].fillet( config["TestPoint_box_rounding"] )
    TA.add_ref( rectangle ).movex(0).movey(0.5*config["TestPoint_gap"])
    TA.add_ref( rectangle ).mirror(p1 = (0, 0), p2 = (200, 0)).movex(0).movey(-0.5*config["TestPoint_gap"])
    TA.center = (0, 0)  
    TA_squid = pg.copy(TA)  
    TA_squid.add_ref(JJ_squid)
    TA_squid.movex(4*config["TestPoint_box_width"])
    TA.add_ref(JJ)
    TA.add_ref(TA_squid)
    TA.center = (0,0)
    
    move_x = config["TestPoint_pos_x"]*0.5*config["Frame_size_width"]
    move_y = config["TestPoint_pos_y"]*0.5*config["Frame_size_height"]   
    TA.move([move_x, move_y])
    TA = pg.union(TA, layer = config["TestPoint_layer"])     
    chipdesign.add_ref(TA)

    # chipdesign.add_ref(FM)
    return chipdesign, devicelist


# def chipdesign_TcSample(config, param_x = None, param_y = None, x = None, y = None, only_frame = False):
@chipdesign_wrapper
def chipdesign_TcSample(config, chipdesign, devicelist):

    # init_chipdesign(config, param_x, param_y, x, y)

    # chipdesign = Device('chipdesign')

    # # Frame
    # FM = device_Frame(config)
    # # chipdesign.add_ref(FM)

    # if only_frame or ((param_x is not None and x is None) or (param_y is not None and y is None)):
    #     chipdesign.add_ref(FM)
    #     return chipdesign

    # Feed line
    FL = device_FeedLine(config)
    chipdesign.add_ref(FL.device)

    # Corner points
    CP = device_CornerPoints(config)
    chipdesign.add_ref(CP)

    # Resonator
    R = []
    for i, resonator_config in enumerate(config["Resonator_devices"]):
        R.append( device_Resonator(config, **resonator_config) )
        R[i].rotate(resonator_config["angle"])

        if config["FeedLine_path_type"] == "straight":
            R[i].xmin = FL.device.x + 0.5*config["LaunchPad_trace_width"] + config["LaunchPad_trace_gap_width"] + resonator_config["feedline_gap"]
            R[i].y = resonator_config["y"]
        elif config["FeedLine_path_type"] == "manual":
            R[i].xmin = config["FeedLine_path_points"][resonator_config["x_pathpoint"]][0] + 0.5*config["LaunchPad_trace_width"] + config["LaunchPad_trace_gap_width"] + resonator_config["feedline_gap"]
            R[i].y = resonator_config["y"]
        elif config["FeedLine_path_type"] == "extrude":
            sys.exit("Currently I don't know how to extract the right position to place the resonators...")
        chipdesign.add_ref(R[i].device)

    devicelist = [
        dict(device = FL, name = "FeedLine"),
        dict(device = R[0], name = "Resonator1"),    
        dict(device = R[1], name = "Resonator2")
    ]

    # chipdesign.add_ref(FM)
    return chipdesign, devicelist