```python
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
```