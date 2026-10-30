# The A3 training scene

The training course, `a3_train`: three windows in the net, registered to the
Vicon frame and stored in it. Load it with `SceneBundle.load("scene")`;
`manifest.json` lists its files.

| file | what it is |
|---|---|
| `a3_train_cleaned_splat.splat` | the Gaussian splat: what the camera sees |
| `a3_train_esdf.npy` | distance field: clearance in metres at any point; scores collisions |
| `a3_train_flight_pc.ply` | the same geometry as a point cloud, for plots |
| `gates.json` | the windows, in the order flown: for checking and scoring only |
| `task.json` | the start, Vicon metres |
| `policy_contract.json` | the scene's camera; `make_contract.py` builds yours from it |
| `vicon_transform.json` | Vicon metres → splat units: here only a scale |
| `anchors.json` | the floor markers the registration was fitted to |
| `splat_frame.json` | where the scale came from |
| `capture_transforms.json`, `nerfstudio_config.yml` | the capture's cameras and training config, for the GPU renderer |

The registration is accurate to 11.7 mm RMS in the floor plane. `check()`
reports no errors or warnings.
