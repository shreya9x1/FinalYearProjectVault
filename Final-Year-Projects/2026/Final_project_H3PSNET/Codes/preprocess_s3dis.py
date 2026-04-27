"""
Preprocess S3DIS Area folders: merge per-room pointcloud with per-object annotation files
and write per-room labeled files with 7 columns: X Y Z R G B LABEL

Usage (PowerShell):
python preprocess_s3dis.py --src "<Area_5_path>" --dst "<output_path>" --room "conferenceRoom_1" --tol 1e-3

If --room is omitted the script will process all room files under src.
"""
import os
import argparse
import numpy as np
from pathlib import Path

# Try to import a fast nearest-neighbor implementation
try:
    from scipy.spatial import cKDTree as KDTree
    KD_IMPLEMENTATION = 'scipy'
except Exception:
    try:
        from sklearn.neighbors import NearestNeighbors
        KD_IMPLEMENTATION = 'sklearn'
    except Exception:
        KD_IMPLEMENTATION = None

CLASS_MAP = {
    'ceiling': 0,
    'floor': 1,
    'wall': 2,
    'beam': 3,
    'column': 4,
    'window': 5,
    'door': 6,
    'table': 7,
    'chair': 8,
    'sofa': 9,
    'bookcase': 10,
    'board': 11,
    'clutter': 12,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Path to Area_X directory')
    p.add_argument('--dst', required=True, help='Destination directory for labeled room files')
    p.add_argument('--room', default=None, help='Optional: process a single room name (folder name)')
    p.add_argument('--tol', type=float, default=1e-3, help='Tolerance for matching coordinates')
    return p.parse_args()


def build_kdtree(points):
    if KD_IMPLEMENTATION == 'scipy':
        return KDTree(points)
    elif KD_IMPLEMENTATION == 'sklearn':
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points)
        return nn
    else:
        raise RuntimeError('No KDTree implementation available. Please install scipy or scikit-learn.')


def query_tree(tree, query_pts):
    if KD_IMPLEMENTATION == 'scipy':
        dists, idxs = tree.query(query_pts, k=1)
        return idxs, dists
    elif KD_IMPLEMENTATION == 'sklearn':
        dists, idxs = tree.kneighbors(query_pts, return_distance=True)
        return idxs[:, 0], dists[:, 0]
    else:
        raise RuntimeError('No KDTree implementation available. Please install scipy or scikit-learn.')


def process_room(room_file, ann_dir, out_file, tol=1e-3):
    # Load room points
    data = np.loadtxt(room_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise RuntimeError(f"Room file {room_file} doesn't have at least 6 columns")

    xyz = data[:, 0:3]
    rgb = data[:, 3:6]
    labels = np.full((xyz.shape[0],), CLASS_MAP['clutter'], dtype=int)

    # Build KD-tree over room XYZ
    tree = build_kdtree(xyz)

    # Iterate annotation files
    if not os.path.isdir(ann_dir):
        print(f"No Annotations folder for {room_file}; writing all-clutter labels")
    else:
        for ann in os.listdir(ann_dir):
            if not ann.lower().endswith('.txt'):
                continue
            ann_path = os.path.join(ann_dir, ann)
            try:
                ann_data = np.loadtxt(ann_path)
            except Exception as e:
                print(f"  Skipping annotation {ann_path}: failed to load ({e})")
                continue
            if ann_data.ndim == 1:
                ann_data = ann_data.reshape(1, -1)
            if ann_data.shape[1] < 3:
                print(f"  Skipping annotation {ann_path}: not enough columns")
                continue
            ann_xyz = ann_data[:, 0:3]
            # parse label name from filename, e.g., chair_1.txt -> chair
            label_name = Path(ann).stem.split('_')[0]
            label_idx = CLASS_MAP.get(label_name.lower(), CLASS_MAP['clutter'])

            idxs, dists = query_tree(tree, ann_xyz)
            # idxs may be shape (M,) or (M,1)
            for i_room, dist in zip(np.atleast_1d(idxs), np.atleast_1d(dists)):
                if dist <= tol:
                    labels[int(i_room)] = label_idx

    # Compose output array
    out = np.concatenate([xyz, rgb, labels.reshape(-1, 1).astype(float)], axis=1)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savetxt(out_file, out, fmt='%f %f %f %d %d %d %d')
    print(f"Wrote labeled file: {out_file}  (N={out.shape[0]})")


def main():
    args = parse_args()
    src = args.src
    dst = args.dst
    tol = args.tol

    if KD_IMPLEMENTATION is None:
        print('Warning: neither scipy nor scikit-learn found. Install scipy for faster processing.')

    # Walk rooms in src: pick .txt files that are NOT inside an 'Annotations' folder
    rooms = []
    for root, _, files in os.walk(src):
        # skip any files inside an Annotations directory
        is_annotations = any(part.lower() == 'annotations' for part in Path(root).parts)
        if is_annotations:
            continue
        for f in files:
            if f.endswith('.txt'):
                rooms.append(os.path.join(root, f))

    if args.room:
        rooms = [r for r in rooms if args.room in r]
        if not rooms:
            print(f"No room matching '{args.room}' found under {src}")
            return

    print(f"Processing {len(rooms)} room(s) from {src} -> {dst}")
    for room in rooms:
        room_dir = os.path.dirname(room)
        room_name = Path(room).stem
        ann_dir = os.path.join(room_dir, 'Annotations')
        rel_dir = os.path.relpath(room_dir, src)
        out_dir = os.path.join(dst, rel_dir)
        out_file = os.path.join(out_dir, room_name + '_labeled.txt')
        try:
            process_room(room, ann_dir, out_file, tol=tol)
        except Exception as e:
            print(f"Failed to process {room}: {e}")

if __name__ == '__main__':
    main()
