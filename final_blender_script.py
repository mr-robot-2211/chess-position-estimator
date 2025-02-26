import bpy
import random
import math
from mathutils import Vector,Matrix

# Set render engine to Cycles
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

# Use GPU for rendering
bpy.context.scene.cycles.device = 'GPU'

# Ensure to select the GPU device if available
if bpy.context.scene.cycles.device == 'GPU':
    for i, device in enumerate(bpy.context.preferences.addons['cycles'].preferences.devices):
        if device.type == 'CUDA':  # For NVIDIA GPUs
            device.use = True
        elif device.type == 'OPENCL':  # For AMD GPUs
            device.use = True

# Remove all lights
for light in bpy.data.lights:
    bpy.data.lights.remove(light)
# Disable ambient occlusion and reflections in Eevee
bpy.context.scene.eevee.use_gtao = False
bpy.context.scene.eevee.use_ssr = False

## Set Cycles bounce settings for consistent lighting if using Cycles
#if bpy.context.scene.render.engine == 'CYCLES':
#    bpy.context.scene.cycles.sampling_pattern = 'SOBOL'
#    bpy.context.scene.cycles.use_denoising = False
#    bpy.context.scene.cycles.max_bounces = 1
#    bpy.context.scene.cycles.diffuse_bounces = 1
#    bpy.context.scene.cycles.glossy_bounces = 1
#    bpy.context.scene.cycles.transparent_max_bounces = 1
#    bpy.context.scene.cycles.transmission_bounces = 1

# Set up basic world ambient light
world = bpy.context.scene.world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs[0].default_value = (0.8, 0.8, 0.8, 1)  # Set light gray ambient light
    bg_node.inputs[1].default_value = 0.2  # Set ambient light strength

# Camera clipping settings
camera = bpy.data.objects.get('Camera')
if camera:
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000

# Set camera exposure and view settings for consistency
bpy.context.scene.view_settings.exposure = 1.0
bpy.context.scene.view_settings.look = 'None'

# Set output to PNG with no compression
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.compression = 0

# Reference to chess piece objects
pieces = ['Cube', 'Cube.001', 'Cube.002', 'Cube.003', 'Cube.004', 'Cube.005',
          'Cube.006', 'Cube.007', 'Plane', 'Plane.001', 'Plane.002', 'Plane.003', 
          'Plane.004', 'Plane.005', 'Plane.006', 'Plane.007', 'Plane.008', 'Plane.009',
          'Plane.010', 'Plane.011', 'Plane.012', 'Plane.013', 'Plane.014', 'Plane.015', 
          'Plane.016', 'Plane.017', 'Plane.018', 'Plane.019', 'Plane.020', 'Plane.021', 
          'Plane.022', 'Plane.024']

piece_objects = {piece: bpy.data.objects[piece] for piece in pieces}

# Define chessboard grid: 8x8 grid
board_size = 8
square_size = 2.0  # Each square is 2 Blender units
piece_z_offset = 0  # Offset to ensure pieces are above the board

def set_origin_to_bottom(obj):
    """Set the origin of the object to the bottom of its bounding box."""
    min_z = min([(obj.matrix_world @ Vector(corner))[2] for corner in obj.bound_box])
    obj.location[2] -= min_z

for piece in pieces:
    set_origin_to_bottom(bpy.data.objects[piece])

def align_piece_origin(obj):
    """Ensure the piece's origin is at its base."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='BOUNDS')
    
    min_z = min([(obj.matrix_world @ Vector(corner))[2] for corner in obj.bound_box])
    obj.location[2] -= min_z
    obj.location[2] += piece_z_offset

def randomize_pieces():
    available_positions = random.sample(range(board_size * board_size), len(pieces))
    
    for index in available_positions:
        row = index // board_size
        col = index % board_size
        
        x_position = (col - (board_size / 2)) * square_size +1
        y_position = (row - (board_size / 2)) * square_size +1
        
        obj = piece_objects[pieces[index % len(pieces)]]
        
        obj.location = (x_position, y_position, 0)
        obj.rotation_euler = (0, 0, random.uniform(0, 6.28))  # Randomize rotation
        
        align_piece_origin(obj)

    change_lighting()
    change_camera_angle()

def change_lighting():
    """Set up a balanced, consistent lighting setup for the scene."""
    # Clear existing lights
    lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
    for light in lights:
        bpy.data.objects.remove(light, do_unlink=True)

    # Create a sun light for directional lighting
    sun_light = bpy.data.objects.new("Sun Light", bpy.data.lights.new(name="Sun", type='SUN'))
    sun_light.data.energy = 4.0  # Adjust energy for balanced brightness
    sun_light.data.angle = 0.3  # Small angle for softer shadows

    # Set a fixed rotation to ensure consistent shadow direction
    sun_light.rotation_euler = (math.radians(45), math.radians(0), math.radians(45))
    
    # Link the sun light to the scene collection
    bpy.context.collection.objects.link(sun_light)
    
    # Set consistent ambient lighting in the world settings
    world = bpy.context.scene.world
    world.use_nodes = True
    background_node = world.node_tree.nodes.get("Background")
    if background_node:
        background_node.inputs[0].default_value = (0.8, 0.8, 0.8, 1)  # Light gray color for ambient light
        background_node.inputs[1].default_value = 0.2  # Ambient intensity to soften shadows slightly


def change_camera_angle():
    """Change the camera position and orientation to focus on the origin (0, 0, 0), ensuring the entire board is visible."""
    camera = bpy.data.objects.get('Camera')
    
    if camera:
        # Randomly choose between two types of orientations
        if random.choice([True, False]):
            # Regular overhead view with a random angle
            z = random.uniform(60, 80)  # Increased height for greater coverage

            # Randomly choose x and y from the specified ranges, ensuring they are farther away
            x = random.choice([random.uniform(-60, -50), random.uniform(50, 60)])  # Extended ranges
            y = random.choice([random.uniform(-60, -50), random.uniform(50, 60)])  # Extended ranges

            camera.location = (x, y, z)

        else:
            # Edge-parallel view (close to the board's edge) but farther
            z = random.uniform(40, 50)  # Increased height to capture more of the board

            # Position the camera at a distance that ensures the board is covered, farther from the origin
            x = random.choice([-square_size * board_size - 30, square_size * board_size + 30])  # Extended distance
            y = random.uniform(-square_size * board_size / 2 - 25, square_size * board_size / 2 + 25)  # Increased range

            camera.location = (x, y, z)

        # Calculate direction vector to the origin
        direction = Vector((0, 0, 0)) - camera.location
        direction.normalize()

        # Rotation matrix to align camera to the origin
        up = Vector((0, 0, 1))
        right = direction.cross(up).normalized()
        up_corrected = right.cross(direction).normalized()
        
        rot_matrix = Matrix((right, up_corrected, -direction)).transposed()
        camera.rotation_euler = rot_matrix.to_euler()

        # Add slight random rotation to avoid perfectly parallel lines
        camera.rotation_euler.z += random.uniform(-0.05, 0.05)

        # Set the lens value to a moderate level to avoid distortion
        camera.data.lens = 50  # Keeping a moderate focal length to minimize distortion

        # Clear any existing constraints to avoid conflicts
        camera.constraints.clear()

# Call the function to change the camera angle
change_camera_angle()

        
def render_chessboard(image_count):
    """Render images of the chessboard with different configurations."""
    for i in range(image_count):
        randomize_pieces()
        
        # Set render file path and name
        bpy.context.scene.render.filepath = f"/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/images-rendered/chessboard_image_{i:03d}.png"
        bpy.ops.render.render(write_still=True)

# Call the function to render a set of images
render_chessboard(image_count=100)  # Test with a reduced number of images
