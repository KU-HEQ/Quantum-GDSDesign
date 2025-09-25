import numpy as np
import math, pprint
import matplotlib.pyplot as plt
from scipy import constants as const
from phidl import quickplot as qp
from phidl import Device
from phidl import CrossSection
from phidl import Path
import phidl.geometry as pg
import phidl.routing as pr
import phidl.path as pp
from functions import *
from BaseDevice import *

finger_layer = 1
box_layer = 2

def resolve_from_string(path: str, scope=None):

    """Returns value from string
    resolve_from_string( "PF[0]" )        -> PF[0]
    resolve_from_string( "PF[0].device" ) -> PF[0].device
    """

    if scope is None:
        scope = locals()

    def resolve_token(obj, token: str, mode: str):
        """トークンを解決（インデックス付きにも対応）"""
        # "name[0]" や "name[1][2]" に対応
        while "[" in token and token.endswith("]"):
            name, idx = token[:-1].split("[", 1)
            if name:   # 例: "PF[0]" → "PF" と "0"
                obj = getattr(obj, name) if hasattr(obj, name) else obj[name]
            obj = obj[int(idx)]
            token = ""  # name 部分を消してループ抜け
        if token:  # 残りが属性名なら getattr
            if mode == "attribute":
                obj = getattr(obj, token)
            elif mode == "object":
                obj = obj[token]
        return obj

    # トップレベルのオブジェクトを取得
    first, *rest = path.split(".")
    obj = resolve_token(scope, first, "object")

    # 属性やさらにネストしたインデックスを辿る
    for token in rest:
        obj = resolve_token(obj, token, "attribute")

    return obj

def make_Path(config,
              resonator_straight1 = 240, 
              resonator_straight2 = 290, 
              resonator_straight3 = 475, # determines the inductive coupling
              resonator_straight4 = 1400, 
              n_step = 3,
              side = False ):

    P = Path()
    left180_turn = pp.arc(radius = config["Resonator_radius"], angle = 180)
    right180_turn = pp.arc(radius = config["Resonator_radius"], angle = -180)
    # left_turn = pp.euler(radius = resonator_radius, angle = 90)
    # right_turn = pp.euler(radius = resonator_radius, angle = -90)
    left_turn = pp.arc(radius = config["Resonator_radius"], angle = 90)
    right_turn = pp.arc(radius = config["Resonator_radius"], angle = -90)
    straight1 = pp.straight(length = resonator_straight1)
    straight2 = pp.straight(length = resonator_straight2)
    straight3 = pp.straight(length = resonator_straight3)
    straight4 = pp.straight(length = resonator_straight4)
    straight5 = pp.straight(length = 250)

    path_list = []
    if side:
        path_list = [
            straight5,
            right_turn,
        ]

    path_list.extend([
        straight4,
        right_turn,
        straight3,
        right180_turn
    ])

    for i in range(n_step):
        if i % 2 == 0:
            turn = left180_turn
        else:
            turn = right180_turn
        path_list.extend([
            straight2,
            turn
        ])
    path_list.extend([straight1])
    P.append(path_list)
    
    return P


def device_Wafer(config):
    wafer = Device('wafer')
    wafer_radius = 0.5 * config["Wafer_inch"] * 25.4 * 1e3 # inch to um
    circle = pg.circle(radius = wafer_radius, angle_resolution = 2.5, layer = config["Wafer_layer"])
    inv_circle = pg.invert(circle, border = 7000, precision = 1e-6, layer = config["Wafer_layer"])
    wafer.add_ref( inv_circle )
    return wafer

def device_Frame(config):
    FM=Device('frame')
    rectangle = pg.rectangle((config["Frame_size_width"] - 2*config["Frame_width"], config["Frame_size_height"] - 2*config["Frame_width"]), config["Frame_layer"])
    FM.add_ref( pg.invert(rectangle, border = config["Frame_width"], precision = 1e-6, layer = config["Frame_layer"]) )
    FM.center = (0, 0)
    return FM

class device_ShortToGround(BaseDevice):
    def __init__(self, config):
        
        super().__init__("short")

        # LP oriented in x direction (x = length, y = width)
        components = {}
        pocket_width = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"]
        components["short"] = pg.rectangle(size = (1, config["LaunchPad_trace_width"])).movey(-0.5*config["LaunchPad_trace_width"])
        components["shortgap"] = pg.rectangle(size = (1, pocket_width)).movey(-0.5*pocket_width)
        components["short_device"] = boolean_with_ports(components["shortgap"], components["short"], "not", layer = config["LaunchPad_layer"])
        components["short_pocket"] = boolean_with_ports(components["shortgap"], components["short"], "or", layer = config["LaunchPad_layer"])
      
        components["short_device"] = self.device.add_ref( components["short_device"] )
        components["short_pocket"] = self.pocket.add_ref( components["short_pocket"] )            

        self.device.add_port(name = 'out', midpoint = [0, 0.], width = pocket_width, orientation = 180)
        self.pocket.add_port(name = 'out', midpoint = [0, 0.], width = pocket_width, orientation = 180)        
        self.center = (0,0)

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = config["LaunchPad_layer"]) )

class device_OpenToGround(BaseDevice):
    def __init__(self, config):
        
        super().__init__("open")

        # LP oriented in x direction (x = length, y = width)
        components = {}
        pocket_width = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"]
        components["open"] = pg.rectangle(size = (1, config["LaunchPad_trace_width"])).movey(-0.5*config["LaunchPad_trace_width"])
        components["opengap"] = pg.rectangle(size = (1 + config["LaunchPad_trace_gap_width"], pocket_width)).movey(-0.5*pocket_width)
        components["open_device"] = boolean_with_ports(components["opengap"], components["open"], "not", layer = config["LaunchPad_layer"])
        components["open_pocket"] = boolean_with_ports(components["opengap"], components["open"], "or", layer = config["LaunchPad_layer"])
      
        components["open_device"] = self.device.add_ref( components["open_device"] )
        components["open_pocket"] = self.pocket.add_ref( components["open_pocket"] )            

        self.device.add_port(name = 'out', midpoint = [0, 0.], width = pocket_width, orientation = 180)
        self.pocket.add_port(name = 'out', midpoint = [0, 0.], width = pocket_width, orientation = 180)        
        self.center = (0,0)

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = config["LaunchPad_layer"]) )

class device_LaunchPad(BaseDevice):
    def __init__(self, config):
        
        super().__init__("launchpad")

        # LP oriented in x direction (x = length, y = width)
        components = {}
        components["pad"] = pg.rectangle(size = (config["LaunchPad_pad_length"],config["LaunchPad_pad_width"])).movey(-0.5*config["LaunchPad_pad_width"])
        components["pad"].add_port(name = 'connect', midpoint = [0., 0.], width = config["LaunchPad_pad_gap_width"], orientation = 180)
        components["padgap"] = pg.rectangle(size = (config["LaunchPad_pad_gap_length"], config["LaunchPad_pad_gap_width"])).movey(-0.5*config["LaunchPad_pad_gap_width"])
        components["pad_device"] = boolean_with_ports(components["padgap"], components["pad"], "not", layer = config["LaunchPad_layer"])
        components["pad_pocket"] = boolean_with_ports(components["padgap"], components["pad"], "or", layer = config["LaunchPad_layer"])
        components["pad_pocket"].add_port(
            name = f'LaunchPad{self.id}_{str(config["LaunchPad_pad_gap_length"] - config["LaunchPad_pad_length"])}', 
            midpoint = [config["LaunchPad_pad_gap_length"], 0.], 
            width = config["LaunchPad_pad_width"], 
            orientation = 180
        )   
        #components["pad_pocket"].add_port(name = f'LaunchPad{self.id}_{str(config["LaunchPad_pad_gap_length"] - config["LaunchPad_pad_length"])}', midpoint = [config["LaunchPad_pad_gap_length"], 0.], width = config["LaunchPad_pad_width"], orientation = 180)

        components["trace"]    = pg.taper(length = config["LaunchPad_trace_length"], width1 = config["LaunchPad_pad_width"], width2 = config["LaunchPad_trace_width"], port = None, layer = 0)
        components["trace"].add_port(name = 'connect', midpoint = [0., 0.], width = config["LaunchPad_pad_gap_width"], orientation = 180)        
        components["tracegap"] = pg.taper(length = config["LaunchPad_trace_length"], width1 = config["LaunchPad_pad_gap_width"], width2 = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], port = None, layer = config["LaunchPad_layer"])
        components["trace_device"] = boolean_with_ports(components["tracegap"], components["trace"], "not", layer = config["LaunchPad_layer"])
        components["trace_pocket"] = boolean_with_ports(components["tracegap"], components["trace"], "or", layer = config["LaunchPad_layer"])        

        components["trace_device"] = self.device.add_ref( components["trace_device"] )
        components["pad_device"] = self.device.add_ref( components["pad_device"] )
        components["trace_pocket"] = self.pocket.add_ref( components["trace_pocket"] )
        components["pad_pocket"] = self.pocket.add_ref( components["pad_pocket"] )        

        components["trace_device"].connect(port = 'connect', destination = components["pad_device"].ports['connect'])
        components["trace_pocket"].connect(port = 'connect', destination = components["pad_pocket"].ports['connect'])        

        self.device.add_port(name = 'out', midpoint = [-config["LaunchPad_trace_length"], 0.], width = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], orientation = 180)
        self.pocket.add_port(name = 'out', midpoint = [-config["LaunchPad_trace_length"], 0.], width = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], orientation = 180)     
        self.center = (0,0)
        self.xmin = 0

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = config["LaunchPad_layer"]) )

class device_Pad(BaseDevice):
    def __init__(self, config):
        super().__init__("PAD")    

        rectangle_up = pg.rectangle(( config["Pad_width"], config["Pad_height"]), config["Pad_layer"])
        rectangle_up.polygons[0].fillet( config["Pad_rounding"] )
        rectangle_up.movex(-0.5*config["Pad_width"])
        #rectangle_up.add_port(name = 'Junction_up', midpoint = [0., 0], width = 10, orientation = -90)
        rectangle_up.movey(0.5*config["Pad_gap"])
        self.metal.add_ref( rectangle_up )

        rectangle_down = pg.rectangle(( config["Pad_width"], config["Pad_height"]), config["Pad_layer"])
        rectangle_down.polygons[0].fillet( config["Pad_rounding"] )
        rectangle_down.mirror(p1 = (0, 0), p2 = (200, 0))
        rectangle_down.movex(-0.5*config["Pad_width"])
        #rectangle_down.add_port(name = f'LaunchPad{self.id}_{str(Pad_gap)}', midpoint = [0, 0.], width = Pad_gap, orientation = 90)
        #rectangle_down.add_port(name = 'Junction_down', midpoint = [0., 0], width = 10, orientation = 90)
        rectangle_down.movey(-0.5*config["Pad_gap"])
        self.metal.add_ref( rectangle_down )
        # self.device.center = (0, 0)
        
        pocket_width = 100
        pocket = pg.rectangle(( config["Pad_width"]+2*pocket_width, 2*config["Pad_height"]+config["Pad_gap"]+2*pocket_width), config["Pad_layer"])
        pocket.center = (0,0)
        pocket.add_port(
            name = f'LaunchPad{self.id}_{str(config["Pad_gap"])}', 
            midpoint = [0, -0.5*config["Pad_gap"]], 
            width = config["Pad_gap"], 
            orientation = 90
        )   
        self.pocket.add_ref(pocket)

        self.device.add_ref( boolean_with_ports(self.pocket, self.metal, "not", layer = config["Pad_layer"]) )

class device_PurcellFilter(BaseDevice):
    def __init__(self, 
                 config,
                 pad_gap,
                 trace_length,
                 n_step,
                 finger_width,
                 finger_length,
                 cap_gap,
                 **kwargs
                 ):
        super().__init__("purcellfilter")    

        
        components = {}
        # Make middle area (pad)
        pad_length = finger_length + cap_gap
        pad_width = n_step * finger_width + (n_step - 1) * cap_gap
        pad_gap_width =  pad_width + 2 * pad_gap
        components["pad_pocket"] = pg.rectangle(size = (pad_length, pad_gap_width))
        components["pad_pocket"].add_port(name = 'connect_in', midpoint = [0., 0.5*pad_gap_width], width = pad_gap_width, orientation = 180)
        components["pad_pocket"].add_port(name = 'connect_out', midpoint = [pad_length, 0.5*pad_gap_width], width = pad_gap_width, orientation = 0)
        components["pad_device"] = pg.copy(components["pad_pocket"])
        for i in range(n_step):
            finger = pg.rectangle(size = (finger_length, finger_width))
            finger.movey(i*(finger_width + cap_gap) + pad_gap)
            if i % 2 == 0:
                finger.movex(cap_gap)
            components["pad_device"] = boolean_with_ports(components["pad_device"], finger, "not", layer = config["PurcellFilter_layer"])

        components["trace"]    = pg.taper(length = trace_length, width1 = pad_width, width2 = config["LaunchPad_trace_width"], port = None, layer = 0)
        rename_port( components["trace"], 1, "connect")
        components["tracegap"] = pg.taper(length = trace_length, width1 = pad_gap_width, width2 = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], port = None, layer = config["PurcellFilter_layer"])
        components["trace_device_in"] = boolean_with_ports(components["tracegap"], components["trace"], "not", layer = config["PurcellFilter_layer"])
        components["trace_pocket"] = boolean_with_ports(components["tracegap"], components["trace"], "or", layer = config["PurcellFilter_layer"])        

        components["trace_device_out"] = pg.copy( components["trace_device_in"] )

        components["trace_device_in"] = self.device.add_ref( components["trace_device_in"] )
        components["trace_device_out"] = self.device.add_ref( components["trace_device_out"] )
        components["pad_device"] = self.device.add_ref( components["pad_device"] )
        components["trace_pocket_in"] = self.pocket.add_ref( components["trace_pocket"] )
        components["trace_pocket_out"] = self.pocket.add_ref( components["trace_pocket"] )
        components["pad_pocket"] = self.pocket.add_ref( components["pad_pocket"] )        

        components["trace_device_in"].connect(port = 'connect', destination = components["pad_device"].ports['connect_in'])
        components["trace_pocket_in"].connect(port = 'connect', destination = components["pad_pocket"].ports['connect_in'])        
        components["trace_device_out"].connect(port = 'connect', destination = components["pad_device"].ports['connect_out'])
        components["trace_pocket_out"].connect(port = 'connect', destination = components["pad_pocket"].ports['connect_out'])   

        self.device.add_port(name = 'in', midpoint = [-trace_length, 0.5*pad_gap_width], width = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], orientation = 180)
        self.device.add_port(name = 'out', midpoint = [pad_length + trace_length, 0.5*pad_gap_width], width = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], orientation = 0)
        self.pocket.add_port(name = 'in', midpoint = [-trace_length, 0.5*pad_gap_width], width = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], orientation = 180)
        self.pocket.add_port(name = 'out', midpoint = [pad_length + trace_length, 0.5*pad_gap_width], width = config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], orientation = 0)
        self.center = (0, 0)
        # self.xmin = 0

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = config["PurcellFilter_layer"]) )

class device_FeedLine_PurcellFilter(BaseDevice):
    def __init__(self, config):
        # make 2 pads
        super().__init__("feedline")

        LP_in = globals()[f"device_{config['FeedLine_input_type']}"](config)
        LP_in.rotate(config["FeedLine_input_angle"]).move(config["FeedLine_input_pos"])

        X_device = CrossSection()
        X_device.add(
            width=config["LaunchPad_trace_gap_width"], 
            offset = 0.5*(config["LaunchPad_trace_width"] + config["LaunchPad_trace_gap_width"]), 
            layer = config["LaunchPad_layer"]
        )
        X_device.add(
            width=config["LaunchPad_trace_gap_width"], 
            offset = -0.5*(config["LaunchPad_trace_width"] + config["LaunchPad_trace_gap_width"]), 
            layer = config["LaunchPad_layer"]
        )

        X_pocket = CrossSection()
        X_pocket.add(
            width=config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], 
            layer = config["LaunchPad_layer"], 
            ports = ('in','out')
        )

        device_ref, metal_ref, pocket_ref = self.add_ref(LP_in)

        LP_out = globals()[f"device_{config['FeedLine_output_type']}"](config)
        LP_out.rotate(config["FeedLine_output_angle"]).move(config["FeedLine_output_pos"])

        PF = []
        for i, purcellfilter_config in enumerate(config["PurcellFilter_devices"]):
            PF.append( device_PurcellFilter( config, **purcellfilter_config ) )
            PF[i].rotate(purcellfilter_config["angle"]).move(purcellfilter_config["pos"])
            self.add_ref(PF[i])

        if config["FeedLine_path_type"] == "extrude":
            P = Path()
            for pathtype, length in config["FeedLine_path_points"]:
                if pathtype == "left":
                    path = pp.arc(radius = length, angle = 90)
                elif pathtype == "right":
                    path = pp.arc(radius = length, angle = -90)
                elif pathtype == "straight":
                    path = pp.straight(length = length)
                P.append(path)

            FeedLine_device = P.extrude(X_device)
            FeedLine_pocket = P.extrude(X_pocket)
            #FeedLine_pocket = P.extrude(LaunchPad_trace_width + 2*LaunchPad_trace_gap_width, layer = LaunchPad_layer)

            ## Get port information
            for port in FeedLine_pocket.get_ports():
                FeedLine_device.add_port(port)
            # FeedLine_device.add_port(name = 'out1', midpoint = [0., 0.], width = LaunchPad_trace_width, orientation = 180)
            # FeedLine_pocket.add_port(name = 'out1', midpoint = [0., 0.], width = LaunchPad_trace_width, orientation = 180)

            FeedLine_device = self.device.add_ref( FeedLine_device )
            FeedLine_device.connect(port = 'in', destination = device_ref.ports['out'])
            
            FeedLine_pocket = self.pocket.add_ref( FeedLine_pocket )
            FeedLine_pocket.connect(port = 'in', destination = pocket_ref.ports['out'])

            device_ref, metal_ref, pocket_ref = self.add_ref(LP_out)

            device_ref.connect(port = "out", destination = FeedLine_device.ports['out'])
            pocket_ref.connect(port = "out", destination = FeedLine_pocket.ports['out'])   
        
        elif config["FeedLine_path_type"] == "manual":

            points = []
            for p in config["FeedLine_path_points"]:
                if isinstance(p, str):
                    points.append( resolve_from_string(p, locals()) )
                else:
                    points.append(p)

            # PurcellFilter の位置を特定
            pf_indices = [i for i, p in enumerate(points) if not isinstance(p, list)]

            segments = []
            ports = []

            # 1. LP_in → 最初の PF
            start_port = LP_in.device.ports['out']
            # end_port = PF[ points[pf_indices[0]] ].device.ports['in']
            end_port = points[pf_indices[0]].device.ports['in']
            ports.append((start_port, end_port))
            segments.append([start_port.midpoint] + points[:pf_indices[0]] + [end_port.midpoint])
            
            # 2. PF[i] → … → PF[i+1]
            for i in range(len(pf_indices)-1):
                start_index = pf_indices[i]
                end_index = pf_indices[i+1]
                # start_port = PF[ points[start_index] ].device.ports['out']
                # end_port = PF[ points[end_index] ].device.ports['in']
                start_port = points[start_index].device.ports['out']
                end_port = points[end_index].device.ports['in']
                start_point = start_port.midpoint
                end_point = end_port.midpoint
                segments.append([start_point] + points[start_index+1:end_index] + [end_point])
                ports.append((start_port, end_port))

            # 3. 最後の PF → LP_out
            # start_port = PF[ points[pf_indices[-1]] ].device.ports['out']
            start_port = points[pf_indices[-1]].device.ports['out']
            end_port = LP_out.device.ports['out']
            segments.append([start_port.midpoint] + points[pf_indices[-1]+1:] + [end_port.midpoint] )
            ports.append((start_port, end_port))

            for ipath, manual_path in enumerate(segments):
                common_args = dict(
                    port1          = ports[ipath][0],
                    port2          = ports[ipath][1],
                    path_type      = "manual",
                    manual_path    = manual_path,
                    radius         = config["FeedLine_path_radius"],
                    smooth_options = {"corner_fun": pp.arc},
                )
                D3 = pr.route_smooth(**common_args, width=X_device)
                D4 = pr.route_smooth(**common_args, layer=config["LaunchPad_layer"])
                    
                self.device.add_ref(D3)
                self.pocket.add_ref(D4)


            self.add_ref(LP_out)

        else:
            common_args = dict(
                port1          = LP_in.device.ports['out'],
                port2          = LP_out.device.ports['out'],
                path_type = config["FeedLine_path_type"], 
                length1 = config["FeedLine_path_length1"],
                length2 = config["FeedLine_path_length2"],
                radius = config["FeedLine_path_radius"],
                smooth_options={'corner_fun': pp.arc}
            )
            D3 = pr.route_smooth(**common_args, width = X_device)
            D4 = pr.route_smooth(**common_args)

            self.device.add_ref(D3)
            self.pocket.add_ref(D4)
            self.add_ref(LP_out)

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = config["LaunchPad_layer"]) )

class device_FeedLine(BaseDevice):
    def __init__(self, config):
        # make 2 pads
        super().__init__("feedline")

        LP_in = globals()[f"device_{config['FeedLine_input_type']}"](config)
        LP_in.rotate(config["FeedLine_input_angle"]).move(config["FeedLine_input_pos"])

        X_device = CrossSection()
        X_device.add(
            width=config["LaunchPad_trace_gap_width"], 
            offset = 0.5*(config["LaunchPad_trace_width"] + config["LaunchPad_trace_gap_width"]), 
            layer = config["LaunchPad_layer"]
        )
        X_device.add(
            width=config["LaunchPad_trace_gap_width"], 
            offset = -0.5*(config["LaunchPad_trace_width"] + config["LaunchPad_trace_gap_width"]), 
            layer = config["LaunchPad_layer"]
        )

        X_pocket = CrossSection()
        X_pocket.add(
            width=config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], 
            layer = config["LaunchPad_layer"], 
            ports = ('in','out')
        )

        device_ref, metal_ref, pocket_ref = self.add_ref(LP_in)

        LP_out = globals()[f"device_{config['FeedLine_output_type']}"](config)
        LP_out.rotate(config["FeedLine_output_angle"]).move(config["FeedLine_output_pos"])

        if config["FeedLine_path_type"] == "extrude":
            P = Path()
            for pathtype, length in config["FeedLine_path_points"]:
                if pathtype == "left":
                    path = pp.arc(radius = length, angle = 90)
                elif pathtype == "right":
                    path = pp.arc(radius = length, angle = -90)
                elif pathtype == "straight":
                    path = pp.straight(length = length)
                P.append(path)

            FeedLine_device = P.extrude(X_device)
            FeedLine_pocket = P.extrude(X_pocket)
            #FeedLine_pocket = P.extrude(LaunchPad_trace_width + 2*LaunchPad_trace_gap_width, layer = LaunchPad_layer)

            ## Get port information
            for port in FeedLine_pocket.get_ports():
                FeedLine_device.add_port(port)
            # FeedLine_device.add_port(name = 'out1', midpoint = [0., 0.], width = LaunchPad_trace_width, orientation = 180)
            # FeedLine_pocket.add_port(name = 'out1', midpoint = [0., 0.], width = LaunchPad_trace_width, orientation = 180)

            FeedLine_device = self.device.add_ref( FeedLine_device )
            FeedLine_device.connect(port = 'in', destination = device_ref.ports['out'])
            
            FeedLine_pocket = self.pocket.add_ref( FeedLine_pocket )
            FeedLine_pocket.connect(port = 'in', destination = pocket_ref.ports['out'])

            device_ref, metal_ref, pocket_ref = self.add_ref(LP_out)

            device_ref.connect(port = "out", destination = FeedLine_device.ports['out'])
            pocket_ref.connect(port = "out", destination = FeedLine_pocket.ports['out'])   
        
        elif config["FeedLine_path_type"] == "manual":
            manual_path = [ LP_in.device.ports['out'].midpoint ] + config["FeedLine_path_points"] +  [ LP_out.device.ports['out'].midpoint ]
            D3 = pr.route_smooth(LP_in.device.ports['out'], 
                                    LP_out.device.ports['out'], 
                                    width = X_device, 
                                    path_type='manual', 
                                    manual_path=manual_path, 
                                    radius = config["FeedLine_path_radius"],
                                    smooth_options={'corner_fun': pp.arc})
            D4 = pr.route_smooth(LP_in.device.ports['out'], 
                                    LP_out.device.ports['out'], 
                                    path_type='manual', 
                                    manual_path=manual_path,
                                    radius = config["FeedLine_path_radius"], 
                                    layer = config["LaunchPad_layer"],
                                    smooth_options={'corner_fun': pp.arc})
            
            self.device.add_ref(D3)
            self.pocket.add_ref(D4)
            self.add_ref(LP_out)

        else:
            D3 = pr.route_smooth(LP_in.device.ports['out'], 
                                 LP_out.device.ports['out'], 
                                 width = X_device, 
                                 path_type = config["FeedLine_path_type"], 
                                 length1 = config["FeedLine_path_length1"],
                                 length2 = config["FeedLine_path_length2"],
                                 radius = config["FeedLine_path_radius"],
                                 smooth_options={'corner_fun': pp.arc})
            D4 = pr.route_smooth(LP_in.pocket.ports['out'], 
                                 LP_out.pocket.ports['out'], 
                                 path_type = config["FeedLine_path_type"],
                                 length1 = config["FeedLine_path_length1"],
                                 length2 = config["FeedLine_path_length2"],
                                 radius = config["FeedLine_path_radius"],
                                 smooth_options={'corner_fun': pp.arc})

            self.device.add_ref(D3)
            self.pocket.add_ref(D4)
            self.add_ref(LP_out)

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = config["LaunchPad_layer"]) )


class device_EntangleLine(BaseDevice):
    def __init__(self, config):
        # make 2 pads
        super().__init__("entangleline")

        X = CrossSection()
        line_width = 10
        line_gap_width = 6
        X.add(width= line_gap_width, offset = 0.5*(line_width + line_gap_width), layer = 4)
        X.add(width= line_gap_width, offset = -0.5*(line_width + line_gap_width), layer = 4)
        config["width"] = X
        self.device = pr.route_smooth( **config )

        config["width"] = line_width + 2*line_gap_width
        self.pocket = pr.route_smooth( **config )

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = 4) )

class device_DCLine(BaseDevice):
    def __init__(self, config):
        super().__init__("DCLine")

        LP = device_LaunchPad(config)
        LP.move((1650, -1300))
        device_ref, metal_ref, pocket_ref = self.add_ref(LP)

        P = Path()
        left_turn = pp.arc(radius = config["DCLine_radius"], angle = 90)
        right_turn = pp.arc(radius = config["DCLine_radius"], angle = -90)
        straight1 = pp.straight(length = 235)
        straight2 = pp.straight(length = 805)
        straight3 = pp.straight(length = 2973)
        straight4 = pp.straight(length = 2960)

        P.append([
            straight1,
            left_turn,
            straight2,
            right_turn,
            straight3,
            right_turn,
            straight4,
        ])

        X = CrossSection()
        # X.add(width=LaunchPad_trace_width, offset = 0., layer = 1)
        X.add(width=config["LaunchPad_trace_gap_width"], offset = 0.5*(config["LaunchPad_trace_width"] + config["LaunchPad_trace_gap_width"]), layer = config["LaunchPad_layer"])
        X.add(width=config["LaunchPad_trace_gap_width"], offset = -0.5*(config["LaunchPad_trace_width"] + config["LaunchPad_trace_gap_width"]), layer = config["LaunchPad_layer"])

        DCLine_device = P.extrude(X)
        DCLine_pocket = P.extrude(config["LaunchPad_trace_width"] + 2*config["LaunchPad_trace_gap_width"], layer = config["LaunchPad_layer"])

        DCLine_device.add_port(name = 'out1', midpoint = [0., 0.], width = config["LaunchPad_trace_width"], orientation = 180)
        DCLine_pocket.add_port(name = 'out1', midpoint = [0., 0.], width = config["LaunchPad_trace_width"], orientation = 180)

        DCLine_device = self.device.add_ref( DCLine_device )
        DCLine_device.connect(port = 'out1', destination = device_ref.ports['out'])
        
        DCLine_pocket = self.pocket.add_ref( DCLine_pocket )
        DCLine_pocket.connect(port = 'out1', destination = pocket_ref.ports['out'])

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = config["LaunchPad_layer"]) )

def device_CornerPoints(config):
    CP = Device("CornerPoints")
    rectangle = pg.rectangle( (config["CornerPoint_width"], config["CornerPoint_width"]), layer = config["CornerPoint_layer"])
    CornerPoint = pg.invert(rectangle, border = config["CornerPoint_gap_width"], precision = 1e-6, layer = config["CornerPoint_layer"])
    for center in config["CornerPoint_pos"]:
        cp = CP.add_ref( CornerPoint )
        cp.center = center
    return CP

def device_TestAreas(config, DCLine = False):

    if DCLine:
        point_pos = [(-1493.5, -1181.413), (-1493.5, -1831.408), (-843.5, -1181.413), (-843.5, -1831.408)]
    else:
        point_pos = [(-1743.5, -1231.413), (-1743.5, -1881.408), (-1093.5, -1231.413), (-1093.5, -1881.408)]

    TPs = Device("TestPoints")
    TP = Device("TestPoint")   
    box = pg.bbox([(-0.5*config["TestPoint_box_width"],-config["TestPoint_box_length"]),(0.5*config["TestPoint_box_width"],0)], layer = config["TestPoint_layer"])
    box = pg.invert(box, border = config["TestPoint_box_gap_width"], precision = 1e-6, layer = config["TestPoint_layer"])

    stub = pg.bbox([(-0.5*config["TestPoint_stub_width"], 0),(0.5*config["TestPoint_stub_width"], config["TestPoint_stub_length"])], layer = config["TestPoint_layer"])
    box = pg.boolean(box, stub, 'not', layer = config["TestPoint_layer"])
    TP.add_ref(box)  

    polpoints = [
        ( -0.5*config["TestPoint_stub_width"]-config["TestPoint_stub_gap_width"], config["TestPoint_box_gap_width"]                 ),
        ( -0.5*config["TestPoint_stub_width"]-config["TestPoint_stub_gap_width"], config["TestPoint_stub_length"] + config["TestPoint_stub_gap_length"] ),
        (  0.5*config["TestPoint_stub_width"]+config["TestPoint_stub_gap_width"], config["TestPoint_stub_length"] + config["TestPoint_stub_gap_length"] ),
        (  0.5*config["TestPoint_stub_width"]+config["TestPoint_stub_gap_width"], config["TestPoint_box_gap_width"]                 ),
        (  0.5*config["TestPoint_stub_width"]                         , config["TestPoint_box_gap_width"]                 ),
        (  0.5*config["TestPoint_stub_width"]                         , config["TestPoint_stub_length"]                   ),
        ( -0.5*config["TestPoint_stub_width"]                         , config["TestPoint_stub_length"]                   ),
        ( -0.5*config["TestPoint_stub_width"]                         , config["TestPoint_box_gap_width"]                 ),
    ]
    TP.add_polygon(polpoints)
    TP = pg.union(TP, by_layer = False, layer = config["TestPoint_layer"])

    for center in point_pos:
        tp = TPs.add_ref( TP )
        tp.xmin = center[0] - 0.5*config["TestPoint_box_width"] - config["TestPoint_box_gap_width"]
        tp.ymin = center[1] - 0.5*config["TestPoint_box_length"] - config["TestPoint_box_gap_width"]

    return TPs

def device_TestBoxes(DCLine = False):

    if DCLine:
        point_pos = [(-1493.5, -1181.413), (-1493.5, -1831.408), (-1493.5, -1500.0), (-843.5, -1181.413), (-843.5, -1831.408), (-843.5, -1500.0)]
    else:
        point_pos = [(-1743.5, -1231.413), (-1743.5, -1881.408), (-1743.5, -1550), (-1093.5, -1231.413), (-1093.5, -1881.408), (-1093.5, -1550)]

    TBXs = Device("TestBoxes")
    TBX = Device("TestBox")
    box_size = 45
    box1 = pg.rectangle((box_size, box_size), layer = 1)
    box2 = pg.copy_layer(box1, 1, 2)
    TBX.add_ref(box1)
    TBX.add_ref(box2)

    for center in point_pos:
        tbx = TBXs.add_ref( TBX )
        tbx.center = center

    return TBXs

class device_Resonator(BaseDevice):
    def __init__(self, 
                 config,
                 resonator_straight1 = 240, 
                 resonator_straight2 = 290, 
                 resonator_straight3 = 475, # determines the inductive coupling
                 resonator_straight4 = 1400, 
                 n_step = 3,
                 norm_to_length = None,
                 norm_to_frequency = None,
                 transmon = True, 
                 side = False, 
                 flip = False,
                 mirror = False, 
                 entangle = False, 
                 print_length = False, 
                 plot_curvature = False,
                 **kwargs
                 ):
        # make 2 pads
        super().__init__("resonator")

        # Create a blank CrossSection
        X = CrossSection()

        # Add a a few "sections" to the cross-section
        X.add(width=config["Resonator_gap_width"], offset = 0.5*(config["Resonator_width"] + config["Resonator_gap_width"]), layer = config["Resonator_layer"])
        X.add(width=config["Resonator_gap_width"], offset = -0.5*(config["Resonator_width"] + config["Resonator_gap_width"]), layer = config["Resonator_layer"])
        
        # Combine the Path and the CrossSection
        P = make_Path(
                 config,
                 resonator_straight1 = resonator_straight1, 
                 resonator_straight2 = resonator_straight2, 
                 resonator_straight3 = resonator_straight3,
                 resonator_straight4 = resonator_straight4,
                 n_step = n_step,
                 side = side 
        )
        if norm_to_length or norm_to_frequency:
            if norm_to_length and norm_to_frequency:
                raise ValueError("Do not set both norm_to_length and norm_to_frequency")
            elif norm_to_frequency:
                norm_to_length = calculate_resonator_length(frequency = norm_to_frequency, material = config["Wafer_material"])
            norm_factor = float( (norm_to_length - (3 + 2*n_step)*(math.pi/2)*config["Resonator_radius"]) /(P.length() - (3 + 2*n_step)*(math.pi/2)*config["Resonator_radius"]) )
            resonator_straight1 = norm_factor * resonator_straight1
            resonator_straight2 = norm_factor * resonator_straight2
            resonator_straight3 = norm_factor * resonator_straight3
            resonator_straight4 = norm_factor * resonator_straight4       
            P = make_Path(
                 config,
                 resonator_straight1 = resonator_straight1, 
                 resonator_straight2 = resonator_straight2, 
                 resonator_straight3 = resonator_straight3,
                 resonator_straight4 = resonator_straight4,
                 n_step = n_step,
                 side = side 
            )                             

        device = P.extrude(X)
        pocket = P.extrude(config["Resonator_width"] + 2*config["Resonator_gap_width"], layer = config["Resonator_layer"])
        device.add_port(name = 'out', midpoint = [0., 0.], width = config["Resonator_width"], orientation = 180)
        pocket.add_port(name = 'out', midpoint = [0., 0.], width = config["Resonator_width"], orientation = 180)
        device = self.device.add_ref(device)
        pocket = self.pocket.add_ref(pocket)        
        self.rotate(90)
        self.movex(-(resonator_straight1 + config["Resonator_radius"]))

        if transmon:
            # capacitor (resonator -> qubit)
            cap_gap1 = 15
            cap_width = 50
            cap_length = 10
            stub_width = config["Resonator_width"]
            stub_length = 2*config["Resonator_width"]

            cap = pg.tee(size = (cap_width,cap_length), stub_size = (stub_width,stub_length), taper_type = 'fillet', layer = 4)
            if entangle:
                cap_entangle = pg.copy( cap )
                cap_entangle.rotate(180)
            line = pg.bbox([(-0.5*stub_width, -config["Resonator_pad_length"]),(0.5*stub_width, 0)])
            cap = pg.boolean(cap, line, 'or', layer = 4)

            # capacitor (qubit)
            cap_gap2 = 16
            cap_width2 = 540
            cap_length2 = 50
            cap_qubit_up = pg.compass_multi(size = (cap_width2, cap_length2), ports = {'N':3,'S':3}, layer = 0)
            cap_qubit_down = pg.compass_multi(size = (cap_width2, cap_length2), ports = {'N':3,'S':3}, layer = 0)
            cap_qubit_up.add_port(name = 'Junction_up', midpoint = [0., -0.5*cap_length2], width = 10, orientation = -90)
            cap_qubit_down.add_port(name = 'Junction_down', midpoint = [0., +0.5*cap_length2], width = 10, orientation = 90)
            cap_qubit_down.ymin = cap.ymax + cap_gap1
            cap_qubit_up.ymin = cap_qubit_down.ymax + cap_gap2
            if entangle:
                cap_entangle.ymin = cap_qubit_up.ymax + cap_gap1
            # Subtract from pad
            pad = pg.bbox([(-0.5*config["Resonator_pad_width"], -0.5*config["Resonator_pad_length"]),(0.5*config["Resonator_pad_width"], 0.5*config["Resonator_pad_length"])])
            pad.movey(cap_length + cap_gap1 + cap_length2 + 0.5*cap_gap2)

            # pad.add_port(name = 'out', midpoint = [0., -stub_width], width = stub_width, orientation = 270)
            pad.add_port(name = 'out', midpoint = [0., -cap_length], width = stub_width, orientation = 270)
            pad.add_port(name = 'entangle', midpoint = [0., 2*(cap_length+cap_gap1+cap_length2)+cap_gap2+stub_length], width = stub_width, orientation = 90)

            for port in cap_qubit_up.get_ports() + cap_qubit_down.get_ports():
                if "Junction" in port.name:
                    pad.add_port(name = port.name, midpoint = port.midpoint, width = port.width, orientation = port.orientation)

            pad_pocket = extract_with_ports(pad, [0])
            pad_device = boolean_with_ports(pad, cap, "not", layer = 4)
            pad_device = boolean_with_ports(pad_device, cap_qubit_up, "not", layer = 4)    
            pad_device = boolean_with_ports(pad_device, cap_qubit_down, "not", layer = 4)        
            if entangle:
                pad_device = boolean_with_ports(pad_device, cap_entangle, "not", layer = 4)        
        
            # Quickplot the resulting Device
            pad_device = self.device.add_ref(pad_device)
            pad_pocket = self.pocket.add_ref(pad_pocket)
            #waveguide_device = Resonator.add_ref(waveguide_device)

            pad_device.connect(port = 'out', destination = device.ports['out'])
            pad_pocket.connect(port = 'out', destination = pocket.ports['out'])            

            if mirror: # flip at pad center
                if side:
                    self.mirror(p1 = (-10, pad_device.center[1]), p2 = (10, pad_device.center[1]) )
                else:
                    self.mirror(p1 = (pad_device.center[0], -10), p2 = (pad_device.center[0], 10) )
            if flip: # flip at pad center
                if not side:
                    self.mirror(p1 = (-10, pad_device.center[1]), p2 = (10, pad_device.center[1]) )
                else:
                    self.mirror(p1 = (pad_device.center[0], -10), p2 = (pad_device.center[0], 10) )
        
        else:
            #waveguide_device = Resonator.add_ref(waveguide_device)
            
            # Add short ground
            short_ground = pg.rectangle(size=(config["Resonator_width"] + 2 * config["Resonator_gap_width"], config["Resonator_gap_width"]), layer = config["Resonator_layer"])
            short_ground.movex(-short_ground.center[0])
            short_ground.add_port(name = 'out', midpoint = [0, 0], width = config["Resonator_width"], orientation = 270)
            short_ground_device = self.device.add_ref(short_ground)
            short_ground_pocket = self.pocket.add_ref(short_ground)            
            short_ground_device.connect(port="out", destination=device.ports['out']) 
            short_ground_pocket.connect(port="out", destination=pocket.ports['out'])             

            if mirror: # flip at waveguide center
                if side:
                    self.mirror(p1 = (-10, self.device.center[1]), p2 = (10, self.device.center[1]) )
                else:
                    self.mirror(p1 = (self.device.center[0], -10), p2 = (self.device.center[0], 10) )
            if flip: # flip at waveguide center
                if not side:
                    self.mirror(p1 = (-10, self.device.center[1]), p2 = (10, self.device.center[1]) )
                else:
                    self.mirror(p1 = (self.device.center[0], -10), p2 = (self.device.center[0], 10) )

        if print_length:
            print(f"Length : {P.length()} [um]")
        if plot_curvature:
            s, K = P.curvature()
            plt.plot(s, K, ".-")
            plt.xlabel("Position along curve (arc length)")
            plt.ylabel("Curvature")

        self.metal.add_ref( boolean_with_ports(self.pocket, self.device, "not", layer = config["Resonator_layer"]) )

# def device_JJ( config, JJ_bridge_width = 1.0, JJ_finger_width = 0.2, JJ_type = "manhattan", JJ_squid = False, JJ_bandage = True, JJ_photolitho = False):
def device_JJ( config ):
    JJ=Device('JJ')
    JJ_half=Device('JJ_half')

    if config["JJ_photolitho"]:
        if (config["JJ_type"] == "mh" or config["JJ_type"] == "manhattan") and config["JJ_photolitho"]:

            finger_width_outer = 2.0
            finger_length_outer = 0.6*config["Pad_gap"]
            finger_length_inner = 0.6*config["Pad_gap"]

            # make finger
            finger_outer = pg.rectangle((finger_width_outer, finger_length_outer), finger_layer)
            # finger_outer.polygons[0].fillet( config["JJ_finger_rounding"] )
            finger_outer.movex(-finger_outer.center[0])
            finger_outer.add_port(name = 'out', midpoint = [0, config["JJ_finger_rounding"]], width = finger_width_outer, orientation = 270)
            
            finger_inner = pg.rectangle((config["JJ_finger_width"], finger_length_inner), finger_layer)
            finger_inner.movex(-finger_inner.center[0])
            finger_inner.add_port(name = 'in', midpoint = [0, finger_length_inner], width = config["JJ_finger_width"], orientation = 90)

            finger_inner1 = JJ_half.add_ref( finger_inner ).rotate(45)
            finger_outer1 = JJ_half.add_ref( finger_outer )
            finger_outer1.center[0] = 0
            finger_outer1.movex(-0.75/math.sqrt(2)*finger_length_inner)
            finger_outer1.movey(+0.4/math.sqrt(2)*finger_length_inner)
            JJ_half.movey(-0.1/math.sqrt(2)*finger_length_inner*float(config["JJ_finger_width"]/0.5))
            #finger_outer1.connect(port = 'out', destination = finger_inner1.ports['in'])
            
            # if config["JJ_squid"]:
            #     finger_inner2 = JJ_half.add_ref( finger_inner ).rotate(-45).movey(-0.2/math.sqrt(2)*finger_length_inner)
            #     finger_outer2 = JJ_half.add_ref( finger_outer )
            #     finger_outer2.connect(port = 'out', destination = finger_inner2.ports['in'])

            JJ_half = pg.union(JJ_half)
            JJ_half.polygons[0].fillet( config["JJ_finger_rounding"] )
            JJ_half = pg.union(JJ_half)

            JJ.add_ref( JJ_half )
            JJ.add_ref( pg.copy(JJ_half).mirror(p1 = (-5, 0), p2 = (5, 0)) ) 
            JJ.center = (0,0)

            # Remove corner in JJ
            JJ = pg.union(JJ)
            JJ.polygons[0].fillet( config["JJ_rounding"] )
            JJ = pg.union(JJ)        
            
            if config["JJ_squid"]:
                JJ.add_ref( pg.copy(JJ).movex(-10) )
            JJ.center = (0,0)

            # finger_width_outer1 = 4.0
            # finger_length_outer1 = 0.2*Pad_gap

            # finger_width_outer2 = 2.0
            # finger_length_outer2 = 0.2*Pad_gap

            # finger_width_inner = width
            # finger_length_inner = 0.2*Pad_gap

            # # finger
            # finger_outer1 = pg.rectangle((finger_width_outer1, finger_length_outer1), finger_layer)
            # finger_outer1.movex(-finger_outer1.center[0])
            # finger_outer1.add_port(name = 'out', midpoint = [0, 0], width = finger_width_outer1, orientation = 270)

            # finger_outer2 = pg.rectangle((finger_width_outer2, finger_length_outer2), finger_layer)
            # finger_outer2.movex(-finger_outer2.center[0])
            # finger_outer2.add_port(name = 'out', midpoint = [0, 0], width = finger_width_outer2, orientation = 270)
            # finger_outer2.add_port(name = 'in', midpoint = [0, finger_length_outer2], width = finger_width_outer2, orientation = 90)        

            # finger_inner = pg.rectangle((finger_width_inner, finger_length_inner), finger_layer)
            # finger_inner.movex(-finger_inner.center[0])
            # finger_inner.add_port(name = 'in', midpoint = [0, finger_length_inner], width = finger_width_inner, orientation = 90)

            # finger_outer1 = JJ_half.add_ref( finger_outer1 )
            # finger_outer2 = JJ_half.add_ref( finger_outer2 )
            # finger_inner = JJ_half.add_ref( finger_inner )

            # finger_outer2.connect(port = 'out', destination = finger_inner.ports['in'])
            # finger_outer1.connect(port = 'out', destination = finger_outer2.ports['in'])        

            # JJ.add_ref( JJ_half ).movey(-20)
            # JJ.add_ref( pg.copy(JJ_half).rotate(90) ).movex(20)

        elif (config["JJ_type"] == "dl" or config["JJ_type"] == "dolan") and not config["JJ_bandage"]:

            JJ_finger_up_width = finger_width
            JJ_bridge_width = bridge_width

            finger_up = pg.bbox([(-0.5*JJ_finger_up_width, 0), (0.5*JJ_finger_up_width, JJ_finger_up_length)], JJ_finger_layer)
            finger_up.movey( 0.5*JJ_bridge_width )
            finger_down = pg.bbox([(-0.5*JJ_finger_down_width, -JJ_finger_down_length), (0.5*JJ_finger_down_width, 0)], JJ_finger_layer)
            finger_down.movey( -0.5*JJ_bridge_width )

            pad_box = pg.bbox([(-0.5*JJ_pad_box_width, 0), (0.5*JJ_pad_box_width, JJ_pad_box_length)], JJ_finger_layer)
            pad_box.movey(0.5*JJ_pad_box_gap)
            
            taper = pg.taper(length = JJ_taper_length, width1 = JJ_taper_width1, width2 = JJ_taper_width2, port = None, layer = JJ_finger_layer)
            taper.rotate(90)
            taper.movey( 0.5*JJ_taper_gap )

            finger_up = JJ_half.add_ref( finger_up )
            finger_down = JJ_half.add_ref( finger_down )
            pad_box_up = JJ_half.add_ref( pad_box )        
            pad_box_down = JJ_half.add_ref( pg.copy(pad_box).mirror(p1 = (-5, 0), p2 = (5, 0)) )
            
            JJ.add_ref( JJ_half )
            if config["JJ_squid"]:
                JJ.add_ref( pg.copy(JJ).movex(-10) )
            JJ.center = (0,0)

            if config["JJ_squid"]:
                JJ.add_ref(taper)
                JJ.add_ref( pg.copy(taper).mirror(p1 = (-5, 0), p2 = (5, 0)) )

            ### Old design
            # finger_width = finger_width_var
            # finger_length = 1.5

            # bridge_width = bridge_width_var
            # bridge_length = 2.0

            # bridge_finger_overlay = 0.8
            # bridge_pad_overlay = 0.42

            # pad_box_width = 18
            # pad_box_length = 10
            # pad_triangle_length = 16
            # pad_rounding_radius = 2
            # pad_finger_dist = 4.5

            # pad_inner_width = 2
            # pad_inner_length = 16

            # # make pad
            # pad_box = pg.rectangle((pad_box_width, pad_box_length), finger_layer)
            # pad_box.movex(-pad_box.center[0])
            # pad_box.add_port(name = 'out', midpoint = [0, 0], width = pad_box_width, orientation = 270)
            # pad_triangle = pg.taper(length = pad_triangle_length, width1 = pad_box_width, width2 = 0, port = None, layer = finger_layer)
            # pad_triangle.rotate(-90)
            # pad_outer = pg.boolean(A = pad_box, B = pad_triangle, operation = 'or',  precision = 1e-6, num_divisions = [1,1], layer = finger_layer)
            # pad_outer.add_port(name = 'out', midpoint = [0, -(pad_triangle_length + pad_finger_dist)], width = pad_box_width, orientation = 270)
            # pad_outer.polygons[0].fillet( pad_rounding_radius )
            
            # pad_inner = pg.rectangle((pad_inner_width, pad_inner_length), finger_layer)
            # pad_inner.add_port(name = 'out', midpoint = [0.5*pad_inner_width, 0], width = pad_inner_width, orientation = 270)

            # finger = pg.bbox([(-0.5*finger_width, -finger_length), (0.5*finger_width, 0)], finger_layer)
            # finger.add_port(name = 'finger_bridge', midpoint = [0, 0], width = finger_width, orientation = 90)
            # finger.add_port(name = 'finger_pad', midpoint = [0, -finger_length], width = finger_width, orientation = 270)

            # bridge = pg.rectangle((bridge_width, bridge_length), box_layer)
            # bridge.add_port(name = 'bridge_finger', midpoint = [0.5*bridge_width, bridge_finger_overlay] , width = finger_width, orientation = 270)
            # bridge.add_port(name = 'bridge_pad', midpoint = [0.5*bridge_width, bridge_length - bridge_pad_overlay] , width = finger_width, orientation = 90)        

            # finger = JJ_half.add_ref( finger )
            # bridge = JJ_half.add_ref( bridge )
            # pad_inner_up = JJ_half.add_ref( pad_inner )
            # pad_inner_down = JJ_half.add_ref( pad_inner )         
            # pad_outer_up = JJ_half.add_ref( pad_outer )
            # pad_outer_down = JJ_half.add_ref( pad_outer ) 

            # bridge.connect(port = 'bridge_finger', destination = finger.ports['finger_bridge'])
            # pad_inner_up.connect(port = 'out', destination = bridge.ports['bridge_pad'])
            # pad_inner_down.connect(port = 'out', destination = finger.ports['finger_pad'])
            # pad_outer_up.connect(port = 'out', destination = pad_inner_up.ports['out'])
            # pad_outer_down.connect(port = 'out', destination = pad_inner_down.ports['out'])                             

            # JJ.add_ref( JJ_half )
            # if config["JJ_squid"]:
            #     JJ.add_ref( pg.copy(JJ).movex(-10) )
            # JJ.center = (0,0)

    else:
        if (config["JJ_type"] == "mh" or config["JJ_type"] == "manhattan") and config["JJ_bandage"]:

            box_finger_overlay_outer = 0.68
            box_finger_overlay_inner = 0.18

            box_outer_width = 1.8
            finger_width_outer1 = 0.405
            finger_length_outer1 = 13.7

            finger_width_outer2 = 0.315
            finger_length_outer2 = 3.6

            finger_width_inner1 = 0.315
            finger_length_inner1 = 4.5

            finger_width_inner2 = width
            finger_length_inner2 = 4.2

            box_inner_width = 0.9

            box_outer = pg.rectangle((box_outer_width, box_outer_width), box_layer)
            box_outer.movex(-box_outer.center[0])
            box_outer.add_port(name = 'out', midpoint = [0, box_finger_overlay_outer], width = finger_width_outer1, orientation = 270)
            # rectangle_subtract = pg.rectangle((finger_width_outer1, box_finger_overlay_outer), box_layer)
            # rectangle_subtract.movex(-rectangle_subtract.center[0])
            # box_outer = pg.boolean(A = box_outer, B = rectangle_subtract, operation = 'not', precision = 1e-6, num_divisions = [1,1], layer = box_layer)

            # finger
            finger_outer1 = pg.rectangle((finger_width_outer1, finger_length_outer1), finger_layer)
            finger_outer1.movex(-finger_outer1.center[0])
            finger_outer1.add_port(name = 'in', midpoint = [0, finger_length_outer1], width = finger_width_outer1, orientation = 90)
            finger_outer1.add_port(name = 'out', midpoint = [0, 0], width = finger_width_outer1, orientation = 270)

            finger_outer2 = pg.rectangle((finger_width_outer2, finger_length_outer2), finger_layer)
            finger_outer2.movex(-finger_outer2.center[0])
            finger_outer2.add_port(name = 'in', midpoint = [0, finger_length_outer2], width = finger_width_outer2, orientation = 90)
            finger_outer2.add_port(name = 'out', midpoint = [0, 0], width = finger_width_outer2, orientation = 270)

            finger_inner1 = pg.rectangle((finger_width_inner1, finger_length_inner1), finger_layer)
            finger_inner1.movex(-finger_inner1.center[0])
            finger_inner1.add_port(name = 'in', midpoint = [0, finger_length_inner1], width = finger_width_inner1, orientation = 90)
            finger_inner1.add_port(name = 'out', midpoint = [0, 0], width = finger_width_inner1, orientation = 270)

            finger_inner2 = pg.rectangle((finger_width_inner2, finger_length_inner2), finger_layer)
            finger_inner2.movex(-finger_inner2.center[0])
            finger_inner2.add_port(name = 'in', midpoint = [0, finger_length_inner2], width = finger_width_inner2, orientation = 90)
            finger_inner2.add_port(name = 'out', midpoint = [0, 0], width = finger_width_inner2, orientation = 270)

            # inner box (x 3)
            box_inner = pg.rectangle((box_inner_width, box_inner_width), box_layer)
            box_inner.movex(-box_inner.center[0])
            box_inner.add_port(name = 'out', midpoint = [0, box_finger_overlay_inner], width = finger_width_inner1, orientation = 270)

            box_outer = JJ_half.add_ref( box_outer )
            finger_outer1 = JJ_half.add_ref( finger_outer1 )
            finger_outer2 = JJ_half.add_ref( finger_outer2 )
            box_inner1 = JJ_half.add_ref( box_inner )
            box_inner1.rotate(180)

            box_inner2 = JJ_half.add_ref( box_inner )
            box_inner2.rotate(45)
            box_inner2.center = (-1.5, -13.5)
            finger_inner1 = JJ_half.add_ref( finger_inner1 )
            finger_inner2 = JJ_half.add_ref( finger_inner2 )
            box_inner3 = JJ_half.add_ref( box_inner )

            finger_outer1.connect(port = 'in', destination = box_outer.ports['out'])
            finger_outer2.connect(port = 'in', destination = finger_outer1.ports['out'])
            box_inner1.connect(port = 'out', destination = finger_outer2.ports['out'])

            finger_inner1.connect(port = 'in', destination = box_inner2.ports['out'])
            finger_inner2.connect(port = 'in', destination = finger_inner1.ports['out'])
            box_inner3.connect(port = 'out', destination = finger_inner2.ports['out'])

            JJ.add_ref( JJ_half )
            JJ.add_ref( pg.copy(JJ_half).mirror(p1 = (-5, -18), p2 = (5, -18)) )
            if config["JJ_squid"]:
                JJ.add_ref( pg.copy(JJ).movex(-10) )
            JJ.center = (0,0)

        elif (config["JJ_type"] == "mh" or config["JJ_type"] == "manhattan") and not config["JJ_bandage"]:
            pad_box_width = 18
            pad_box_length = 10
            pad_triangle_length = 16
            pad_rounding_radius = 2

            box_width = 1.2
            box_finger_overlay = 0.24
            pad_finger_overlay = 2

            # make pad
            pad_box = pg.rectangle((pad_box_width, pad_box_length), finger_layer)
            pad_box.movex(-pad_box.center[0])
            pad_box.add_port(name = 'out', midpoint = [0, 0], width = pad_box_width, orientation = 270)
            pad_triangle = pg.taper(length = pad_triangle_length, width1 = pad_box_width, width2 = 0, port = None, layer = finger_layer)
            pad_triangle.add_port(name = 'out1', midpoint = [0.95*(pad_triangle_length-pad_rounding_radius), 0], width = pad_box_width, orientation = 45)
            pad_triangle.add_port(name = 'out2', midpoint = [0.95*(pad_triangle_length-pad_rounding_radius), 0], width = pad_box_width, orientation = -45)
            pad_box = JJ_half.add_ref( pad_box )
            pad_triangle = JJ_half.add_ref( pad_triangle )
            pad_triangle.connect(port = 1, destination = pad_box.ports['out'])
            JJ_half = pg.union(JJ_half, by_layer = False, layer = finger_layer)
            JJ_half.polygons[0].fillet( pad_rounding_radius )

            # make finger
            finger = pg.taper(length = config["JJ_finger_length"] + pad_finger_overlay, width1 = config["JJ_finger_width"], width2 = config["JJ_finger_width"], port = None, layer = finger_layer)
            finger.add_port(name = 'out1', midpoint = [pad_finger_overlay, 0], width = config["JJ_finger_width"], orientation = 180)
            finger1 = JJ_half.add_ref( finger )
            finger1.connect(port = 'out1', destination = pad_triangle.ports['out1'])
            if config["JJ_squid"]:
                finger2 = JJ_half.add_ref( finger )        
                finger2.connect(port = 'out1', destination = pad_triangle.ports['out2'])

            # make box
            box = pg.rectangle((box_width, box_width), box_layer)
            box.movex(-box.center[0])
            box.add_port(name = 'out', midpoint = [0, box_finger_overlay], width = config["JJ_finger_width"], orientation = 270)
            #box1 = JJ_half.add_ref( box )
            #box1.connect(port = 'out', destination = finger1.ports[2])
            # if config["JJ_squid"]:
            #     box2 = JJ_half.add_ref( box )
            #     #box2.connect(port = 'out', destination = finger2.ports[2])

            JJ.add_ref( JJ_half )
            JJ.add_ref( pg.copy(JJ_half).mirror(p1 = (-5, -18), p2 = (5, -18)) ) 
            JJ.center = (0,0)

            # Make additional finger for bilayer sample
            finger_horizontal = pg.rectangle((0.5*config["JJ_finger_length"], config["JJ_finger_width"]), finger_layer)
            finger_horizontal.center = (0, 0)
            finger_horizontal1 = JJ.add_ref( finger_horizontal )
            finger_horizontal1.movex(0.45*config["JJ_finger_length"])
            if config["JJ_squid"]:
                finger_horizontal2 = JJ.add_ref( finger_horizontal )
                finger_horizontal2.movex(-0.45*config["JJ_finger_length"])

        if (config["JJ_type"] == "dl" or config["JJ_type"] == "dolan") and config["JJ_bandage"]:
            finger_width = finger_width_var
            finger_length = 1.5

            bridge_width = bridge_width_var
            bridge_length = 2.0

            bridge_finger_overlay = 0.8
            bridge_pad_overlay = 0.42

            pad_width = 2
            pad_length = 16

            bandage_gap = 0.65
            bandage1_width = 0.2
            bandage1_length = 3.0
            bandage2_width = 0.45
            bandage2_length = 3.2

            nbandage = 4

            finger = pg.bbox([(-0.5*finger_width, -finger_length), (0.5*finger_width, 0)], finger_layer)
            finger.add_port(name = 'finger_bridge', midpoint = [0, 0], width = finger_width, orientation = 90)
            finger.add_port(name = 'finger_pad', midpoint = [0, -finger_length], width = finger_width, orientation = 270)

            bridge = pg.rectangle((bridge_width, bridge_length), box_layer)
            bridge.add_port(name = 'bridge_finger', midpoint = [0.5*bridge_width, bridge_finger_overlay] , width = finger_width, orientation = 270)
            bridge.add_port(name = 'bridge_pad', midpoint = [0.5*bridge_width, bridge_length - bridge_pad_overlay] , width = finger_width, orientation = 90)        

            pad = pg.rectangle((pad_width, pad_length), finger_layer)
            pad.add_port(name = 'out', midpoint = [0.5*pad_width, 0], width = pad_width, orientation = 270)
            for i in range(nbandage):
                pad.add_port(name = f'bandage{i}', midpoint = [0.5*pad_width, pad_length - (i+1)*bandage_gap - (i + 0.5)*bandage2_width], width = bandage2_width, orientation = 0)

            bandage1 = pg.rectangle((bandage1_width, bandage1_length), finger_layer)
            bandage2 = pg.rectangle((bandage2_width, bandage2_length), box_layer)
            bandage1.add_port(name = 'bandage1_pad', midpoint = [0.5*bandage1_width, 0], width = bandage1_width, orientation = 270)
            bandage2.add_port(name = 'bandage2_pad', midpoint = [0.5*bandage2_width, 0], width = bandage2_width, orientation = 270)

            finger = JJ_half.add_ref( finger )
            bridge = JJ_half.add_ref( bridge )
            pad_up = JJ_half.add_ref( pad )
            pad_down = JJ_half.add_ref( pad )
            pad_down.mirror(p1 = (0,0), p2 = (0, -5)) 

            bandage1_up = []
            bandage2_up = []        
            bandage1_down = []
            bandage2_down = []        
            for i in range(nbandage):
                bandage1_up.append( JJ_half.add_ref(bandage1) )
                bandage2_up.append( JJ_half.add_ref(bandage2) )            
                bandage1_down.append( JJ_half.add_ref(bandage1) )
                bandage2_down.append( JJ_half.add_ref(bandage2) )                        

            bridge.connect(port = 'bridge_finger', destination = finger.ports['finger_bridge'])
            pad_up.connect(port = 'out', destination = bridge.ports['bridge_pad'])
            pad_down.connect(port = 'out', destination = finger.ports['finger_pad'])

            for i in range(nbandage):
                bandage1_up[i].connect(port = 'bandage1_pad', destination = pad_up.ports[f'bandage{i}'])
                bandage2_up[i].connect(port = 'bandage2_pad', destination = pad_up.ports[f'bandage{i}'])            
                bandage1_down[i].connect(port = 'bandage1_pad', destination = pad_down.ports[f'bandage{i}'])
                bandage2_down[i].connect(port = 'bandage2_pad', destination = pad_down.ports[f'bandage{i}'])                        

            JJ.add_ref( JJ_half )
            if config["JJ_squid"]:
                JJ.add_ref( pg.copy(JJ).movex(-10) )
            JJ.center = (0,0)

        if (config["JJ_type"] == "dl" or config["JJ_type"] == "dolan") and not config["JJ_bandage"]:

            JJ_finger_up_width = finger_width
            JJ_bridge_width = bridge_width

            finger_up = pg.bbox([(-0.5*JJ_finger_up_width, 0), (0.5*JJ_finger_up_width, JJ_finger_up_length)], JJ_finger_layer)
            finger_up.movey( 0.5*JJ_bridge_width )
            finger_down = pg.bbox([(-0.5*JJ_finger_down_width, -JJ_finger_down_length), (0.5*JJ_finger_down_width, 0)], JJ_finger_layer)
            finger_down.movey( -0.5*JJ_bridge_width )

            pad_box = pg.bbox([(-0.5*JJ_pad_box_width, 0), (0.5*JJ_pad_box_width, JJ_pad_box_length)], JJ_finger_layer)
            pad_box.movey(0.5*JJ_pad_box_gap)
            
            taper = pg.taper(length = JJ_taper_length, width1 = JJ_taper_width1, width2 = JJ_taper_width2, port = None, layer = JJ_finger_layer)
            taper.rotate(90)
            taper.movey( 0.5*JJ_taper_gap )

            finger_up = JJ_half.add_ref( finger_up )
            finger_down = JJ_half.add_ref( finger_down )
            pad_box_up = JJ_half.add_ref( pad_box )        
            pad_box_down = JJ_half.add_ref( pg.copy(pad_box).mirror(p1 = (-5, 0), p2 = (5, 0)) )
            
            JJ.add_ref( JJ_half )
            if config["JJ_squid"]:
                JJ.add_ref( pg.copy(JJ).movex(-10) )
            JJ.center = (0,0)

            if config["JJ_squid"]:
                JJ.add_ref(taper)
                JJ.add_ref( pg.copy(taper).mirror(p1 = (-5, 0), p2 = (5, 0)) )

            ### Old design
            # finger_width = finger_width_var
            # finger_length = 1.5

            # bridge_width = bridge_width_var
            # bridge_length = 2.0

            # bridge_finger_overlay = 0.8
            # bridge_pad_overlay = 0.42

            # pad_box_width = 18
            # pad_box_length = 10
            # pad_triangle_length = 16
            # pad_rounding_radius = 2
            # pad_finger_dist = 4.5

            # pad_inner_width = 2
            # pad_inner_length = 16

            # # make pad
            # pad_box = pg.rectangle((pad_box_width, pad_box_length), finger_layer)
            # pad_box.movex(-pad_box.center[0])
            # pad_box.add_port(name = 'out', midpoint = [0, 0], width = pad_box_width, orientation = 270)
            # pad_triangle = pg.taper(length = pad_triangle_length, width1 = pad_box_width, width2 = 0, port = None, layer = finger_layer)
            # pad_triangle.rotate(-90)
            # pad_outer = pg.boolean(A = pad_box, B = pad_triangle, operation = 'or',  precision = 1e-6, num_divisions = [1,1], layer = finger_layer)
            # pad_outer.add_port(name = 'out', midpoint = [0, -(pad_triangle_length + pad_finger_dist)], width = pad_box_width, orientation = 270)
            # pad_outer.polygons[0].fillet( pad_rounding_radius )
            
            # pad_inner = pg.rectangle((pad_inner_width, pad_inner_length), finger_layer)
            # pad_inner.add_port(name = 'out', midpoint = [0.5*pad_inner_width, 0], width = pad_inner_width, orientation = 270)

            # finger = pg.bbox([(-0.5*finger_width, -finger_length), (0.5*finger_width, 0)], finger_layer)
            # finger.add_port(name = 'finger_bridge', midpoint = [0, 0], width = finger_width, orientation = 90)
            # finger.add_port(name = 'finger_pad', midpoint = [0, -finger_length], width = finger_width, orientation = 270)

            # bridge = pg.rectangle((bridge_width, bridge_length), box_layer)
            # bridge.add_port(name = 'bridge_finger', midpoint = [0.5*bridge_width, bridge_finger_overlay] , width = finger_width, orientation = 270)
            # bridge.add_port(name = 'bridge_pad', midpoint = [0.5*bridge_width, bridge_length - bridge_pad_overlay] , width = finger_width, orientation = 90)        

            # finger = JJ_half.add_ref( finger )
            # bridge = JJ_half.add_ref( bridge )
            # pad_inner_up = JJ_half.add_ref( pad_inner )
            # pad_inner_down = JJ_half.add_ref( pad_inner )         
            # pad_outer_up = JJ_half.add_ref( pad_outer )
            # pad_outer_down = JJ_half.add_ref( pad_outer ) 

            # bridge.connect(port = 'bridge_finger', destination = finger.ports['finger_bridge'])
            # pad_inner_up.connect(port = 'out', destination = bridge.ports['bridge_pad'])
            # pad_inner_down.connect(port = 'out', destination = finger.ports['finger_pad'])
            # pad_outer_up.connect(port = 'out', destination = pad_inner_up.ports['out'])
            # pad_outer_down.connect(port = 'out', destination = pad_inner_down.ports['out'])                             

            # JJ.add_ref( JJ_half )
            # if config["JJ_squid"]:
            #     JJ.add_ref( pg.copy(JJ).movex(-10) )
            # JJ.center = (0,0)

    return JJ

def device_EBLine(config):
    EBLine=Device('EBLine')

    box = pg.rectangle((config["EBLine_box_width"], config["EBLine_box_width"]), config["EBLine_box_layer"])
    box.movex(-box.center[0])
    box.add_port(name = 'out', midpoint = [0, config["EBLine_box_overlay"]], width = config["EBLine_finger_outer_width"], orientation = 270)

    # finger
    finger_outer = pg.rectangle((config["EBLine_finger_outer_width"], config["EBLine_finger_outer_length"]), config["EBLine_finger_layer"])
    finger_outer.movex(-finger_outer.center[0])
    finger_outer.add_port(name = 'in', midpoint = [0, config["EBLine_finger_outer_length"]], width = config["EBLine_finger_outer_width"], orientation = 90)
    finger_outer.add_port(name = 'out', midpoint = [0, 0], width = config["EBLine_finger_outer_width"], orientation = 270)

    finger_inner = pg.rectangle((config["EBLine_finger_inner_width"], config["EBLine_finger_inner_length"]), config["EBLine_finger_layer"])
    finger_inner.movex(-finger_inner.center[0])
    finger_inner.add_port(name = 'in', midpoint = [0, config["EBLine_finger_inner_length"]], width = config["EBLine_finger_inner_width"], orientation = 90)
    finger_inner.add_port(name = 'out', midpoint = [0, 0], width = config["EBLine_finger_inner_width"], orientation = 270)

    box_up = EBLine.add_ref( box )
    box_down = EBLine.add_ref( box )    
    finger_outer_up = EBLine.add_ref( finger_outer )
    finger_outer_down = EBLine.add_ref( finger_outer )
    finger_inner = EBLine.add_ref( finger_inner )

    finger_outer_up.connect(port = 'in', destination = box_up.ports['out'])
    finger_inner.connect(port = 'in', destination = finger_outer.ports['out'])
    finger_outer_down.connect(port = 'in', destination = finger_inner.ports['out'])
    box_down.connect(port = 'out', destination = finger_outer_down.ports['out'])

    EBLine.center = (0,0)
    return EBLine

def device_EBmarkers(marker_pos = [(0,0),(0,38400),(-19200,-28800),(38400,0), (0,-38400),(-19200,-38400),(-38400,0)],layer = 3):
    EBmarkers = Device("EBmarkers")
    EBmarker = Device("EBmarker")
    markers = {}
    tmp1 = pg.bbox([(-5,-20), (5,20)])
    tmp2 = pg.bbox([(-20,-5), (20,5)])
    markers["EBr1"] = pg.boolean(tmp1, tmp2, 'or', layer = layer)
    tmp1 = pg.bbox([(-40,-40), (40,40)])
    tmp2 = pg.bbox([(-30,-30), (30,30)])
    markers["EBr2"] = pg.boolean(tmp1, tmp2, 'not', layer = layer)

    markers["EBf1"] = pg.rectangle((10,10), layer)
    markers["EBf1"].move((-5,-5-200))
    markers["EBf2"] = pg.copy(markers["EBf1"]).rotate(90)
    markers["EBf3"] = pg.copy(markers["EBf1"]).rotate(180)
    markers["EBf4"] = pg.copy(markers["EBf1"]).rotate(270)


    markers["guide1"] = pg.rectangle((10,1850), layer)
    markers["guide1"].move((-5,500))
    markers["guide2"] = pg.copy(markers["guide1"]).rotate(90)
    markers["guide3"] = pg.copy(markers["guide1"]).rotate(180)
    markers["guide4"] = pg.copy(markers["guide1"]).rotate(270)

    # markers["EB_EPFL1"] = pg.rectangle((20,20), layer)
    # markers["EB_EPFL1"].move((-10,-10-2900))
    # markers["EB_EPFL2"] = pg.copy(markers["EB_EPFL1"]).rotate(90)
    # markers["EB_EPFL3"] = pg.copy(markers["EB_EPFL1"]).rotate(180)
    # markers["EB_EPFL4"] = pg.copy(markers["EB_EPFL1"]).rotate(270)

    for key in markers.keys():
        EBmarker.add_ref(markers[key])

    for ii in range(len(marker_pos)):
        marker = EBmarkers.add_ref(EBmarker)
        marker.center = marker_pos[ii]
    return EBmarkers


def device_DicingMarkers(config):
    DicingMarkers = Device("DicingMarkers")
    tmp1 = pg.bbox([
        (-0.5*config["DicingMarker_width"],-0.5*config["DicingMarker_length"]), 
        ( 0.5*config["DicingMarker_width"], 0.5*config["DicingMarker_length"])
    ])
    tmp2 = pg.bbox([
        (-0.5*config["DicingMarker_length"],-0.5*config["DicingMarker_width"]), 
        ( 0.5*config["DicingMarker_length"], 0.5*config["DicingMarker_width"])
    ])    
    marker = pg.boolean(tmp1, tmp2, 'or', layer = config["DicingMarker_layer"])

    DicingMarkers.add_ref(marker)
    return DicingMarkers

def device_Grid(config):
    grid = Device("Grid")
    wafer_radius = 0.5 * config["Wafer_inch"] * 25.4 * 1e3 # inch to um
    device_list_perp = [pg.rectangle(size = (2*config["Frame_width"], 2*wafer_radius), layer = config["Grid_layer"]) for i in range(config["Grid_lines_x"])]
    device_list_horiz = [pg.rectangle(size = (2*wafer_radius, 2*config["Frame_width"]), layer = config["Grid_layer"]) for i in range(config["Grid_lines_y"])]
    grid_perp = pg.grid(device_list_perp,
                spacing = (config["Frame_size_width"] - 2*config["Frame_width"], 0),
                # separation = False,
                shape = (config["Grid_lines_x"],1))
    grid_perp.center = (0, 0)
    grid_horiz = pg.grid(device_list_horiz,
                spacing = (0, config["Frame_size_height"] - 2*config["Frame_width"]),
                # separation = False,
                shape = (1, config["Grid_lines_y"])) 
    grid_horiz.center = (0, 0)        

    if "Grid_sweep_type" in config:
        if config["Grid_sweep_type"] == "gridsweep":
            x = 1
            y = 1
            for i, array in enumerate(config["Grid_sweep_array"]):
                x = x * len(array["x"]) + array["gap_x"] * (len(array["x"]) - 1)
                y = y * len(array["y"]) + array["gap_y"] * (len(array["y"]) - 1)
            print(x, y)
        elif config["Grid_sweep_type"] == "array":
            shape = np.array(config["Grid_sweep_array"], dtype=object).shape
            x = shape[0] + (shape[0] - 1) * config["Grid_sweep_gap_x"]
            y = shape[1] + (shape[1] - 1) * config["Grid_sweep_gap_y"]
        if (x % 2 == 0 and config["Grid_lines_x"] % 2 == 0) or (x % 2 == 1 and config["Grid_lines_x"] % 2 == 1):
            grid_perp.center = (0.5 * config["Frame_size_width"], 0)
        if (y % 2 == 0 and config["Grid_lines_y"] % 2 == 0) or (y % 2 == 1 and config["Grid_lines_y"] % 2 == 1):
            grid_horiz.center = (0, 0.5 * config["Frame_size_height"])

    circle = pg.circle(radius = wafer_radius, angle_resolution = 2.5, layer = 21)
    inv_circle = pg.invert(circle, border = 7000, precision = 1e-6, layer = 21)
    grid_perp = pg.boolean(A = grid_perp, B = inv_circle, operation = 'not', precision = 1e-6,
                num_divisions = [1,1], layer = config["Grid_layer"])
    grid_horiz = pg.boolean(A = grid_horiz, B = inv_circle, operation = 'not', precision = 1e-6,
                num_divisions = [1,1], layer = config["Grid_layer"])    
    grid.add_ref( grid_perp )
    grid.add_ref( grid_horiz )    
    return grid