import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, Gf, Sdf
CAMERA_PATH = "/World/DollyCamera"
TRAVEL_AXIS = "Z"
# World-space start and end positions along the travel axis
START_VALUE = -20000   # e.g. -500 cm
END_VALUE   =  -15000   # e.g. +500 cm
FIXED_AXIS_1_VALUE = 850   # x
FIXED_AXIS_2_VALUE =   250 # y
# Frame range
START_FRAME = 1
END_FRAME   = 1200
# Frames per second (should match your stage's timeCodesPerSecond)
FPS = 60
# Focal length in mm (optional, set None to leave unchanged)
FOCAL_LENGTH_MM = 35.0
# ─────────────────────────────────────────────────────────────────────────────
def _make_translate(axis: str, along: float, f1: float, f2: float) -> Gf.Vec3d:
    """Build a Vec3d given a travel axis and two fixed values."""
    axis = axis.upper()
    if axis == "X":
        return Gf.Vec3d(along, f1, f2)
    elif axis == "Y":
        return Gf.Vec3d(f1, along, f2)
    elif axis == "Z":
        return Gf.Vec3d(f1, f2, along)
    else:
        raise ValueError(f"TRAVEL_AXIS must be 'X', 'Y', or 'Z', got '{axis}'")
def create_dolly_camera():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No stage is open. Open or create a USD stage first.")
    # ── Stage time settings ────────────────────────────────────────────────
    stage.SetStartTimeCode(START_FRAME)
    stage.SetEndTimeCode(END_FRAME)
    stage.SetTimeCodesPerSecond(FPS)
    # ── Get or create the camera prim ─────────────────────────────────────
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not camera_prim.IsValid():
        omni.kit.commands.execute(
            "CreatePrimWithDefaultXform",
            prim_type="Camera",
            prim_path=CAMERA_PATH,
        )
        camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
        print(f"[dolly] Created camera at {CAMERA_PATH}")
    else:
        print(f"[dolly] Using existing camera at {CAMERA_PATH}")
    camera = UsdGeom.Camera(camera_prim)
    # ── Optional: set focal length ─────────────────────────────────────────
    if FOCAL_LENGTH_MM is not None:
        camera.GetFocalLengthAttr().Set(FOCAL_LENGTH_MM)
    # ── Build the xformOp:translate attribute ─────────────────────────────
    xform = UsdGeom.Xformable(camera_prim)
    # Clear any existing xform ops to start clean
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_op.Set(Gf.Vec3d(0.0, 180.0, 0.0))
    start_pos = _make_translate(
        TRAVEL_AXIS, START_VALUE, FIXED_AXIS_1_VALUE, FIXED_AXIS_2_VALUE
    )
    end_pos = _make_translate(
        TRAVEL_AXIS, END_VALUE, FIXED_AXIS_1_VALUE, FIXED_AXIS_2_VALUE
    )
    # Set keyframes
    translate_op.Set(start_pos, Usd.TimeCode(START_FRAME))
    translate_op.Set(end_pos,   Usd.TimeCode(END_FRAME))
    print(
        f"[dolly] Keyframes set:\n"
        f"  frame {START_FRAME:>4}  →  {start_pos}\n"
        f"  frame {END_FRAME:>4}  →  {end_pos}\n"
        f"  axis={TRAVEL_AXIS}, duration={END_FRAME - START_FRAME + 1} frames @ {FPS} fps"
    )
    # ── Save the stage ─────────────────────────────────────────────────────
    omni.usd.get_context().save_stage()
    print("[dolly] Stage saved. Press Play in the timeline to preview.")
if __name__ == "__main__":
    create_dolly_camera()