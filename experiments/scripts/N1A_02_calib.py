# N1A_02_calib -- k calibration for the steering magnitude (see N1A_02_causal_core.py for design + predictions).
N1A_MODE = "calib"
N1A_RUN_LAYERS = [10, 20, 30, 40, 45, 50, 55, 57]
N1A_KS = {"k1": 1.0, "k3": 3.0, "k10": 10.0, "k30": 30.0}
N1A_GENDER = False
exec(open("./experiments/scripts/N1A_02_causal_core.py").read())
