from math import cos, sin, radians
import uuid
from pydantic import BaseModel, Field, conint
from typing import List, Optional, Union


class Wall(BaseModel):
    optimize: int = Field(description="Whether to optimize the position and rotation of the asset", default=0)
    corner1: List[float] = Field(description="2d coordinates of the first corner of this wall")
    corner2: List[float] = Field(description="2d coordinates of the second corner of this wall")
    instance_id: Optional[str] = Field(description="Unique identifier for the wall", default=None)

class AssetInstance(BaseModel):
    optimize: int = Field(description="Whether to optimize the position and rotation of the asset", default=1)
    position: List[float] = Field(description="Position of the asset", default=[0,0,0])
    rotation: List[float] = Field(description="Rotation of the asset in degrees", default=[0,0,0])
    instance_id: Optional[str] = Field(description="Unique identifier for the asset instance", default=None)
    size: Optional[List[float]] = Field(description="Bounding box size of the asset instance", default=0.01)

    # if the position is 2d, add a zero z-coordinate
    def __init__(self, **data):
        if 'position' in data and len(data['position']) == 2:
            data['position'].append(0)
        super().__init__(**data)

    # Method to set size from the parent Assets class if not provided
    def set_size_from_parent(self, parent_size: Optional[List[float]]):
        if self.size is None and parent_size is not None:
            self.size = parent_size


class Assets(BaseModel):
    description: str = Field(description="Description of the asset")
    placements: List[AssetInstance] = Field(description="List of asset instances of this 3D asset", default=None)
    size: Optional[List[float]] = Field(description="Bounding box size of the asset (z-axis up)", default=None)
    onCeiling: bool = Field(description="Whether the asset is on the ceiling", default=False)

    def __getitem__(self, index: int) -> AssetInstance:
        "Allow indexing into placements."
        return self.placements[index]
    
    def __len__(self) -> int:
        "Allow using len() to get the number of placements."
        return len(self.placements)


class Constraint:
    def __init__(self, constraint_name, **params):
        self.constraint_name = constraint_name
        self.params = params
    
    def evaluate(self, assets: list):
        return self.constraint_func(assets, **self.params)


fixed_pointtt = Assets(description="global absolute marks", size=[0.01, 0.01, 0.01], placements=[])
def get_instance_id(asset):
    global fixed_pointtt
    if asset.instance_id is not None:
        return asset.instance_id

    if isinstance(asset, AssetInstance):
        # fine the name of the variable that is an instance of AssetInstance
        for var_name in globals(): 
            var = globals()[var_name]
            if isinstance(var, Assets):
                for instance_idx, placement in enumerate(var.placements):
                    if id(placement) == id(asset):
                        return f"{var_name}_{instance_idx}"

        # new instantiation of AssetInstance, return random instance_id
        instance_index = len(fixed_pointtt.placements)
        fixed_pointtt.placements.append(
            AssetInstance(
                position=asset.position,
                rotation=[0,0,0],
                optimize=False,
                instance_id=f"fixed_pointtt_{instance_index}"
            )
        )
        return fixed_pointtt.placements[-1].instance_id

    if isinstance(asset, Wall):
        var = globals()['walls']
        for instance_idx, wall_instance in enumerate(var):
            if id(wall_instance) == id(asset):
                return f"walls_{instance_idx}"


class ConstraintSolver:
    def __init__(self):
        self.constraints = []

    def handle_fixed_pointtt(self, asset):
        global fixed_pointtt
        if isinstance(asset, tuple):
            instance_index = len(fixed_pointtt.placements)
            fixed_pointtt.placements.append(
                AssetInstance(
                    position=list(asset),
                    rotation=[0,0,0],
                    optimize=False,
                    instance_id=f"fixed_pointtt_{instance_index}"
                )
            )
            return fixed_pointtt.placements[-1]
        return asset
    
    def point_towards(self, asset1: AssetInstance, asset2: Union[AssetInstance, tuple], angle=0):
        asset1.instance_id = get_instance_id(asset1)
        asset2 = self.handle_fixed_pointtt(asset2)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("point_towards", angle=angle),
            [asset1.instance_id, asset2.instance_id]
        ])

    def distance_constraint(self, asset1: AssetInstance, asset2: Union[AssetInstance, tuple], min_distance=0, max_distance=10000, weight=1):
        asset1.instance_id = get_instance_id(asset1)
        asset2 = self.handle_fixed_pointtt(asset2)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("distance_constraint", min_distance=min_distance, max_distance=max_distance, weight=weight),
            [asset1.instance_id, asset2.instance_id]
        ])

    def against_wall(self, asset1: AssetInstance, wall: Wall):
        asset1.instance_id = get_instance_id(asset1)
        wall.instance_id = get_instance_id(wall)
        self.constraints.append([
            Constraint("against_wall"),
            [asset1.instance_id, wall.instance_id]
        ])

    def on_top_of(self, asset1: AssetInstance, asset2: AssetInstance):
        asset1.instance_id = get_instance_id(asset1)
        asset2 = self.handle_fixed_pointtt(asset2)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("on_top_of"),
            [asset1.instance_id, asset2.instance_id]
        ])
    
    def align_with(self, asset1: AssetInstance, asset2: AssetInstance, angle=0):
        asset1.instance_id = get_instance_id(asset1)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("align_with", angle=angle),
            [asset1.instance_id, asset2.instance_id]
        ])

    def align_x(self, asset1: AssetInstance, asset2: AssetInstance):
        """
        Add a constraint that asset1 should have the same x-coordinate as asset2.
        """
        asset1.instance_id = get_instance_id(asset1)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("align_x"),
            [asset1.instance_id, asset2.instance_id]
        ])

    def align_y(self, asset1: AssetInstance, asset2: AssetInstance):
        """
        Add a constraint that asset1 should have the same y-coordinate as asset2.
        """
        asset1.instance_id = get_instance_id(asset1)
        asset2.instance_id = get_instance_id(asset2)
        self.constraints.append([
            Constraint("align_y"),
            [asset1.instance_id, asset2.instance_id]
        ])

    def solve(self):
        pass

        
solver = ConstraintSolver()

# Walls that define the boundary of the scene
walls = [
    Wall(corner1=[0.00, 0.00, 0.00], corner2=[4.00, 0.00, 0.00]),
    Wall(corner1=[4.00, 0.00, 0.00], corner2=[4.00, 5.00, 0.00]),
    Wall(corner1=[4.00, 5.00, 0.00], corner2=[0.00, 5.00, 0.00]),
    Wall(corner1=[0.00, 5.00, 0.00], corner2=[0.00, 0.00, 0.00])
]

# Existing assets placed in the scene:

# New assets to be placed
chair = Assets(description="This is a rattan chair with a curved backrest and armrests, featuring a woven texture on the seating surface and back, supported by a metal frame.", size=[0.58, 0.56, 0.80], placements=[AssetInstance() for _ in range(1)])
bed = Assets(description="A queen-sized bed with a neatly arranged duvet, two fluffed pillows against the headboard, and a clean, smooth appearance.", size=[2.00, 1.76, 1.05], placements=[AssetInstance() for _ in range(1)])
night_stand = Assets(description="A night stand with a single pull-out drawer and an open shelf below. It has a flat top, and the drawer is fitted with a simple horizontal handle.", size=[0.39, 0.50, 0.44], placements=[AssetInstance() for _ in range(2)])
table_lamp = Assets(description="This is a table lamp with a cylindrical base and a matching cylindrical shade, likely designed to provide ambient or task lighting.", size=[0.18, 0.18, 0.45], placements=[AssetInstance() for _ in range(2)])
mat = Assets(description="This is a rectangular mat with a textured diamond pattern, likely used for floor protection or decoration.", size=[0.73, 1.20, 0.02], placements=[AssetInstance() for _ in range(1)])
chalkboard = Assets(description="This object is a rectangular chalkboard with a wooden frame, featuring a black writing surface that shows some faint chalk marks, and a decorative golden mandala design on the reverse side.", size=[0.04, 0.57, 0.80], placements=[AssetInstance() for _ in range(1)])

chair[0].instance_id = 'chair_0'
bed[0].instance_id = 'bed_0'
night_stand[0].instance_id = 'night_stand_0'
night_stand[1].instance_id = 'night_stand_1'
table_lamp[0].instance_id = 'table_lamp_0'
table_lamp[1].instance_id = 'table_lamp_1'
mat[0].instance_id = 'mat_0'
chalkboard[0].instance_id = 'chalkboard_0'
walls[0].instance_id = 'walls_0'
walls[1].instance_id = 'walls_1'
walls[2].instance_id = 'walls_2'
walls[3].instance_id = 'walls_3'

chair[0].position = [2.5872149324509475, 0.8692123621047088, 0.4000000059604645]
chair[0].rotation = [0, 0, 0]
bed[0].position = [1.117101726842558, 2.7178240398773057, 0.5247374176979065]
bed[0].rotation = [0, 0, 0]
night_stand[0].position = [0.11325943969382424, 1.0614167209999026, 0.22210413217544556]
night_stand[0].rotation = [0, 0, 0]
night_stand[1].position = [0.742914366626481, 2.4065848910845005, 0.22210413217544556]
night_stand[1].rotation = [0, 0, 0]
table_lamp[0].position = [2.4910389425903308, 0.16293764061530325, 0.22499997913837433]
table_lamp[0].rotation = [0, 0, 0]
table_lamp[1].position = [1.8222564854723524, 2.0231619020954454, 0.22499997913837433]
table_lamp[1].rotation = [0, 0, 0]
mat[0].position = [2.3826732793976557, 1.1590420740173029, 0.00835350714623928]
mat[0].rotation = [0, 0, 0]
chalkboard[0].position = [0.24433488792658054, 0.9060411183726097, 0.4000000059604645]
chalkboard[0].rotation = [0, 0, 0]
chair[0].position = [2.5872149324509475, 0.8692123621047088, 0.4000000059604645]
chair[0].rotation = [0, 0, 0]
bed[0].position = [1.117101726842558, 2.7178240398773057, 0.5247374176979065]
bed[0].rotation = [0, 0, 0]
night_stand[0].position = [0.11325943969382424, 1.0614167209999026, 0.22210413217544556]
night_stand[0].rotation = [0, 0, 0]
night_stand[1].position = [0.742914366626481, 2.4065848910845005, 0.22210413217544556]
night_stand[1].rotation = [0, 0, 0]
table_lamp[0].position = [2.4910389425903308, 0.16293764061530325, 0.22499997913837433]
table_lamp[0].rotation = [0, 0, 0]
table_lamp[1].position = [1.8222564854723524, 2.0231619020954454, 0.22499997913837433]
table_lamp[1].rotation = [0, 0, 0]
mat[0].position = [2.3826732793976557, 1.1590420740173029, 0.00835350714623928]
mat[0].rotation = [0, 0, 0]
chalkboard[0].position = [0.24433488792658054, 0.9060411183726097, 0.4000000059604645]
chalkboard[0].rotation = [0, 0, 0]

solver.constraints = []

# Define initial positions and rotations for the new assets
bed_position = [2, 3, 0.525]  # Centered in the room against the far wall
chair_position = [1, 4, 0.40]  # Near the front wall, leaving space for walking
night_stand_position = [3.5, 3.5, 0.22]  # To the right of the bed
night_stand2_position = [0.5, 3.5, 0.22]  # To the left of the bed
table_lamp_position = [3.5, 3.5, 0.665]  # On top of the right night stand
table_lamp2_position = [0.5, 3.5, 0.665]  # On top of the left night stand
mat_position = [2, 0.5, 0.01]  # Near the middle of the room
chalkboard_position = [0.02, 2.5, 0.40]  # Against the left wall

# Assign positions and rotations based on the layout criteria
bed.placements[0].position = bed_position
chair.placements[0].position = chair_position
night_stand.placements[0].position = night_stand_position
night_stand.placements[1].position = night_stand2_position
table_lamp.placements[0].position = table_lamp_position
table_lamp.placements[1].position = table_lamp2_position
mat.placements[0].position = mat_position
chalkboard.placements[0].position = chalkboard_position

# Bed Constraints
solver.against_wall(bed[0], walls[2])  # Bed against the back wall
solver.point_towards(bed[0], AssetInstance(position=[2, 2, 0]))  # Bed points towards center

# Chair Constraints
solver.against_wall(chair[0], walls[0])  # Chair against the front wall
solver.align_with(chair[0], bed[0], angle=90)  # Chair aligns perpendicularly to bed

# Night Stand Constraints
solver.against_wall(night_stand[0], walls[2])  # Night stand on the right against back wall
solver.against_wall(night_stand[1], walls[2])  # Night stand on the left against back wall

# Table Lamp Constraints
solver.on_top_of(table_lamp[0], night_stand[0])  # Table lamp on right night stand
solver.on_top_of(table_lamp[1], night_stand[1])  # Table lamp on left night stand

# Mat Constraints
solver.distance_constraint(mat[0], AssetInstance(position=[2, 0, 0]), 0, 0.1, weight=10)  # Mat near the middle front

# Chalkboard Constraints
solver.against_wall(chalkboard[0], walls[0])  # Chalkboard against front wall
solver.align_with(chalkboard[0], bed[0])  # Aligned with bed

# Arrangement focuses on open space, clear paths, and alignment to natural light sources

solver.constraints = []
night_stand[1].position = [0.19370724260807037, 4.754325866699219, 0.2199999988079071]
night_stand[1].rotation = [0, 0, 0.00016351049858332847]
night_stand[1].optimize = 2
chalkboard[0].position = [0.019999980926513672, 2.4751148223876953, 0.4000000059604645]
chalkboard[0].rotation = [0, 0, 4.613870513386347e-08]
chalkboard[0].optimize = 2
bed[0].position = [2.0643131732940674, 1.954538106918335, 0.5249999761581421]
bed[0].rotation = [0, 0, -3.141591798356021]
bed[0].optimize = 2
mat[0].position = [2.0481693744659424, 1.6532530784606934, 0.009999999776482582]
mat[0].rotation = [0, 0, 0.0062831852796656935]
mat[0].optimize = 2
table_lamp[0].position = [3.8454504013061523, 4.451111316680908, 0.6649999618530273]
table_lamp[0].rotation = [0, 0, 0.0062831852796656935]
table_lamp[0].optimize = 2
night_stand[0].position = [3.8046576976776123, 4.663900375366211, 0.2199999988079071]
night_stand[0].rotation = [0, 0, 3.1415892016495044]
night_stand[0].optimize = 2
chair[0].position = [1.4843829870224, 4.710000038146973, 0.4000000059604645]
chair[0].rotation = [0, 0, -1.5707962072928183]
chair[0].optimize = 2
table_lamp[1].position = [0.08937466889619827, 4.5036940574646, 0.6649999618530273]
table_lamp[1].rotation = [0, 0, 0.0062831852796656935]
table_lamp[1].optimize = 2

