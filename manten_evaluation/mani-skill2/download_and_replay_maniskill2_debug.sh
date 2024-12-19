#!/bin/bash


# 1) download demonstrations
# reference: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
mkdir -p data/maniskill2
cd data/maniskill2 || exit
mkdir demos

echo "Downloading demonstrations to $(pwd)/demos"
echo "Downloading assets to $(pwd)/data"
for rigid_task in {"PegInsertionSide-v1","StackCube-v1","PickCube-v1"}
do
python -m mani_skill.utils.download_demo -o ./demos/ "$rigid_task"
python -m mani_skill.utils.download_asset "$rigid_task" -y
done


# 3) replay demonstrations
# reference: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html
# we use `pd_ee_delta_pose` as the target control mode by default
echo "Replaying demonstrations..."
for rigid_task in {"PegInsertionSide-v1","StackCube-v1","PickCube-v1"}
do
for obs_mode in {"rgbd","pointcloud"}
do
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "demos/$rigid_task/motionplanning/trajectory.h5" \
  --save-traj --target-control-mode pd_ee_delta_pose --obs-mode "$obs_mode" --num-procs 10 \
  --count 30
done
done
