# N1A_02_main_b -- causal c_p for layers 30..57 (+ tool 'gender' poles at 30/40/50); see N1A_02_causal_core.py.
N1A_MODE = "main"
N1A_RUN_LAYERS = [30, 40, 45, 50, 55, 57]
N1A_KS = {"kA": 60.0, "kB": 180.0}
N1A_GENDER = True
exec(open("./experiments/scripts/N1A_02_causal_core.py").read())
