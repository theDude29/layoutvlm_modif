import os
import json
import argparse
import numpy as np
import collections
from src.layoutvlm.scene import Scene
from src.layoutvlm.layoutvlm import LayoutVLM
from utils.placement_utils import get_random_placement

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json_file", help="Path to scene JSON file", required=True)
    parser.add_argument("--proximity_points_file", help="Optional text file with 2D points (x y per line) to avoid", default=None)
    parser.add_argument("--proximity_radius", help="Radius around each proximity point to penalize (meters)", type=float, default=1.0)
    parser.add_argument("--proximity_weight", help="Weight of the proximity penalty", type=float, default=10.0)
    parser.add_argument("--save_dir", help="Directory to save results", default="./results/test_run")
    parser.add_argument("--model", help="Model to use for layout generation", default="gpt-4")
    parser.add_argument("--openai_api_key", help="OpenAI API key", required=True)
    parser.add_argument("--asset_dir", help="Directory to load assets from.", default="./objaverse_processed")
    return parser.parse_args()

def prepare_task_assets(task, asset_dir):
    """
    Prepare assets for the task by processing their metadata and annotations.
    This is a minimal version that assumes assets are already downloaded and processed.
    """
    if "layout_criteria" not in task:
        task["layout_criteria"] = "the layout should follow the task description and adhere to common sense"

    all_data = collections.defaultdict(list)
    for original_uid in task["assets"].keys():
        # Remove the idx number from the uid
        uid = '-'.join(original_uid.split('-')[:-1])
        
        # Load asset data
        data_path = os.path.join(asset_dir, uid, "data.json")
        if not os.path.exists(data_path):
            print(f"Warning: Asset data not found for {uid}")
            continue
            
        with open(data_path, "r") as f:
            data = json.load(f)
        data['path'] = os.path.join(asset_dir, uid, f"{uid}.glb")
        all_data[uid].append(data)

    # Process categories and create asset entries
    category_count = collections.defaultdict(int)
    for uid, duplicated_assets in all_data.items():
        category_var_name = duplicated_assets[0]['annotations']['category']
        category_var_name = category_var_name.replace('-', "_").replace(" ", "_").replace("'", "_").replace("/", "_").replace(",", "_").lower()
        category_count[category_var_name] += 1

    task["assets"] = {}
    category_idx = collections.defaultdict(int)
    
    for uid, duplicated_assets in all_data.items():
        category_var_name = duplicated_assets[0]['annotations']['category']
        category_var_name = category_var_name.replace('-', "_").replace(" ", "_").replace("'", "_").replace("/", "_").replace(",", "_").lower()
        category_idx[category_var_name] += 1
        
        for instance_idx, data in enumerate(duplicated_assets):
            # Create category name with suffix if needed
            category_var_name = f"{category_var_name}_{chr(ord('A') + category_idx[category_var_name]-1)}" if category_count[category_var_name] > 1 else category_var_name
            
            # Create instance name
            var_name = f"{category_var_name}_{instance_idx}" if len(duplicated_assets) > 1 else category_var_name
            
            # Create asset entry
            task["assets"][f"{category_var_name}-{instance_idx}"] = {
                "uid": uid,
                "count": len(duplicated_assets),
                "instance_var_name": var_name,
                "asset_var_name": category_var_name,
                "instance_idx": instance_idx,
                "annotations": data["annotations"],
                "category": data["annotations"]["category"],
                'description': data['annotations']['description'],
                'path': data['path'],
                'onCeiling': data['annotations']['onCeiling'],
                'onFloor': data['annotations']['onFloor'],
                'onWall': data['annotations']['onWall'],
                'onObject': data['annotations']['onObject'],
                'frontView': data['annotations']['frontView'],
                'assetMetadata': {
                    "boundingBox": {
                        "x": float(data['assetMetadata']['boundingBox']['y']),  # SWAP x and y
                        "y": float(data['assetMetadata']['boundingBox']['x']),
                        "z": float(data['assetMetadata']['boundingBox']['z'])
                    },
                }
            }

    return task

def main():
    args = parse_args()
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load scene configuration
    with open(args.scene_json_file, 'r') as f:
        scene_config = json.load(f)
    
    # Prepare assets
    scene_config = prepare_task_assets(scene_config, args.asset_dir)
    
    # Initialize constraint solver
    layout_solver = LayoutVLM(
        mode="one_shot",
        save_dir=args.save_dir,
        asset_source="objaverse",  # Default to objaverse
        proximity_points=None,
        proximity_radius=args.proximity_radius,
        proximity_weight=args.proximity_weight
    )

    # If a proximity points file is provided, read it and pass to the solver
    if args.proximity_points_file is not None:
        points = []
        with open(args.proximity_points_file, 'r') as pf:
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append([x, y])
                except ValueError:
                    continue
        layout_solver.proximity_points = points
    
    # Generate layout
    layout = layout_solver.solve(scene_config)
    
    # Save results
    output_path = os.path.join(args.save_dir, 'layout.json')
    with open(output_path, 'w') as f:
        json.dump(layout, f, indent=2)
    
    print(f"Layout generated and saved to {output_path}")

if __name__ == "__main__":
    main() 
