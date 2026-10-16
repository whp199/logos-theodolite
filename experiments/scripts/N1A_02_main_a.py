# N1A_02_main_a -- causal c_p for layers 10, 20 (see N1A_02_causal_core.py for design + predictions).
N1A_MODE = "main"
N1A_RUN_LAYERS = [10, 20]
N1A_KS = {"kA": 60.0, "kB": 180.0}
N1A_GENDER = True
exec(open("./experiments/scripts/N1A_02_causal_core.py").read())
