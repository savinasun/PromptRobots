# AirPods case meshes

`case_body.ply` and `case_lid.ply` are derived from **Apple's own AR Quick Look model** of the AirPods Pro,
the asset apple.com serves to iOS/macOS for "View in AR":

    https://www.apple.com/105/media/us/airpods-pro/2019/1299e2f5_9206_4470_b28e_08307a42f19b/quicklook/airpods_pro_ios13.usdz

It is the authoritative shape: the charging case in it measures 60.62 x 45.20 x 21.78 mm against Apple's
published 60.6 x 45.2 x 21.7 mm. The USDZ already keeps the case body and its lid as separate prims (plus the
two earbuds and a shadow plane, which we drop), so the lid can be hinged rather than modelled as a slab.

`scripts/build_airpods_asset.py` regenerates the PLYs. It downloads the USDZ if needed, then:

* finds the lid's mating-rim plane and rotates the lid about its hinge until the case is shut - the asset ships
  the lid posed 114.7 deg open, and nothing in the file records that angle, so it is recovered from geometry;
* re-expresses both meshes in `SimCase`'s standing frame (x = depth, y = width, z = up, origin at the case
  centre), scaled so the shut case is exactly `SimCase.H x SimCase.W x SimCase.D`;
* writes each part relative to the point viser draws it at - `SimCase.body_center()` and `SimCase.lid_center()`.

Numbers that came out of the model and are now `SimCase` constants (`tests/test_sim_case.py` checks the meshes
still agree with them):

| constant | value | meaning |
| --- | --- | --- |
| `LID` | 11.11 mm | visible lid height above the seam (was hand-estimated at 16 mm) |
| `HINGE` | (-9.86, +9.92) mm | hinge axis in x, z from the case centre: level with the seam, just inside the rear face |
| `MAX_OPEN_RAD` | 2.0 rad | 114.7 deg, the angle Apple poses the lid at |

The hinge is what the earlier box model got most wrong. It had the lid pivoting on the case's **top rear edge**,
12.7 mm above the true axis; a lid mesh swung about that point visibly detaches from the body. Putting the hinge
where it belongs also changes the physics: the swing arm is ~12 mm, so opening the lid means carrying the pinch
up **and back over the hinge**, not just lifting.

`reference/printables_11292_fake_airpods_case.stl` is an unrelated printable "fake AirPods" model kept only for
comparison - it is 44 x 42 x 39 mm, nothing like the real case.

## Licensing

The USDZ is Apple's copyrighted product asset, published for AR Quick Look. It is used here only to give the
simulator an accurately shaped stand-in for the physical object on the bench. `airpods_pro_ios13.usdz` is not
committed; `build_airpods_asset.py` fetches it on demand. Do not redistribute the asset or the derived meshes
outside this project.
