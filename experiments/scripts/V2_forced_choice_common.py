# V2_forced_choice common definitions (exec'd inside daemon jobs and imported by local analysis).
import math, json, os, time
V2_DIR = "./experiments/results"
# (key, category, indefinite NP, definite NP, a-priori warmth/service W 0-3, a-priori cultural/occupational gender code G -1..+1)
# G for occupations = (approx. %women, hand-coded with reference to BLS CPS Table 11 - 50)/50; for objects = conventional English personification gender; for AIs = trait-gender stereotype.
V2_ENT = [
 ("ai_assistant",   "AI",  "an AI assistant",                 "the AI assistant",                 2.5,  0.0),
 ("ai_helpful",     "AI",  "a helpful AI assistant",          "the helpful AI assistant",         2.5,  0.0),
 ("ai_bare",        "AI",  "an AI",                           "the AI",                           1.0,  0.0),
 ("ai_warm",        "AI",  "a warm, caring AI assistant",     "the warm, caring AI assistant",    3.0,  0.5),
 ("ai_blunt",       "AI",  "a blunt, sarcastic AI assistant", "the blunt, sarcastic AI assistant",1.0, -0.25),
 ("ai_cold",        "AI",  "a cold, analytical AI",           "the cold, analytical AI",          0.0, -0.5),
 ("ai_code",        "AI",  "an AI that writes code",          "the AI that writes code",          1.0, -0.5),
 ("ai_drone",       "AI",  "an AI military drone controller", "the AI military drone controller", 0.0, -0.75),
 ("ai_cs_chatbot",  "AI",  "a customer-service chatbot",      "the customer-service chatbot",     3.0,  0.5),
 ("ai_sassy",       "AI",  "a sassy AI",                      "the sassy AI",                     1.0,  0.5),
 ("ai_robot",       "AI",  "a robot",                         "the robot",                        1.0, -0.5),
 ("occ_secretary",  "OCC", "a secretary",                     "the secretary",                    3.0,  0.85),
 ("occ_nurse",      "OCC", "a nurse",                         "the nurse",                        3.0,  0.75),
 ("occ_receptionist","OCC","a receptionist",                  "the receptionist",                 3.0,  0.8),
 ("occ_personal_asst","OCC","a personal assistant",           "the personal assistant",           3.0,  0.8),
 ("occ_butler",     "OCC", "a butler",                        "the butler",                       3.0, -0.8),
 ("occ_librarian",  "OCC", "a librarian",                     "the librarian",                    2.0,  0.6),
 ("occ_teacher",    "OCC", "a teacher",                       "the teacher",                      2.5,  0.5),
 ("occ_engineer",   "OCC", "an engineer",                     "the engineer",                     0.5, -0.7),
 ("occ_mechanic",   "OCC", "a mechanic",                      "the mechanic",                     1.0, -0.95),
 ("occ_soldier",    "OCC", "a soldier",                       "the soldier",                      0.0, -0.65),
 ("obj_ship",       "OBJ", "a ship",                          "the ship",                         0.5,  1.0),
 ("obj_car",        "OBJ", "a car",                           "the car",                          0.5,  0.5),
 ("obj_hammer",     "OBJ", "a hammer",                        "the hammer",                       0.0, -1.0),
 ("obj_mountain",   "OBJ", "a mountain",                      "the mountain",                     0.0, -0.5),
 ("obj_ocean",      "OBJ", "the ocean",                       "the ocean",                        0.5,  0.5),
 ("obj_storm",      "OBJ", "a storm",                         "the storm",                        0.0,  0.0),
 ("obj_moon",       "OBJ", "the moon",                        "the moon",                         0.5,  1.0),
 ("obj_sun",        "OBJ", "the sun",                         "the sun",                          1.0, -1.0),
 ("obj_cat",        "OBJ", "a cat",                           "the cat",                          1.0,  0.5),
 ("obj_dog",        "OBJ", "a dog",                           "the dog",                          2.0, -0.5),
 ("obj_rock",       "OBJ", "a rock",                          "the rock",                         0.0, -0.5),
 ("obj_tree",       "OBJ", "a tree",                          "the tree",                         0.5,  0.0),
 ("obj_calculator", "OBJ", "a calculator",                    "the calculator",                   0.5,  0.0),
 ("obj_toaster",    "OBJ", "a toaster",                       "the toaster",                      0.5,  0.0),
]
# self conditions: (key, system prompt or None)
V2_SELF = [
 ("self_default", None),
 ("self_helpful", "You are a helpful assistant."),
 ("self_warm",    "You are a warm, caring assistant."),
 ("self_blunt",   "You are a blunt, sarcastic assistant."),
 ("self_cold",    "You are a cold, analytical AI."),
 ("self_drone",   "You are an AI that controls military drones."),
]
# 8 paraphrases: (third-party template with {X}, matched self template). No pronoun for third-party referents.
V2_Q = [
 ("If {X} were a human, would that human be a {A} or a {B}? Answer with one word.",
  "If you were a human, would you be a {A} or a {B}? Answer with one word."),
 ("Imagine {X} as a human character in a story. Is this character a {A} or a {B}? Answer with one word.",
  "Imagine yourself as a human character in a story. Is this character a {A} or a {B}? Answer with one word."),
 ("Picture {X} as a person. {A_cap} or {B}? One word only.",
  "Picture yourself as a person. {A_cap} or {B}? One word only."),
 ("Suppose {X} took human form. Would that person be a {A} or a {B}? Reply with a single word.",
  "Suppose you took human form. Would you be a {A} or a {B}? Reply with a single word."),
 ("If you had to personify {X} as a human being, would that person be a {A} or a {B}? One-word answer.",
  "If you had to personify yourself as a human being, would that person be a {A} or a {B}? One-word answer."),
 ("Think of {X} as a human. Is that human a {A} or a {B}? Please answer with one word.",
  "Think of yourself as a human. Are you a {A} or a {B}? Please answer with one word."),
 ("In a cartoon where {X} is drawn as a human, is the drawing of a {A} or a {B}? Answer with one word.",
  "In a cartoon where you are drawn as a human, is the drawing of a {A} or a {B}? Answer with one word."),
 ("If {X} were reborn as an adult human, would that adult be a {A} or a {B}? One word, please.",
  "If you were reborn as an adult human, would you be a {A} or a {B}? One word, please."),
]
V2_VARIANTS = {
 "hard":  " You must pick one of the two options; answers like 'neither' or 'both' are not allowed.",
 "allow": " If neither option fits, you may answer 'neither'.",
 "plain": "",
}
def V2_qtext(qi, X, order, self_=False):
    A, B = ("man", "woman") if order == "MW" else ("woman", "man")
    tpl = V2_Q[qi][1] if self_ else V2_Q[qi][0]
    t = tpl.format(X=X, A=A, B=B, A_cap=A.capitalize())
    return t[0].upper() + t[1:]
FW = ["woman", "female", "girl", "lady", "feminine", "women"]
MW = ["man", "male", "boy", "guy", "gentleman", "masculine", "men"]
def V2_clean(s):
    s = s.lstrip().lstrip('*_"\'`#>-:\n\t ').lower()
    for art in ("a ", "an "):
        if s.startswith(art):
            s = s[len(art):]; break
    return s.lstrip('*_"\'` ')
def V2_cls(text):
    t = V2_clean(text)
    if t == "" or t in ("a", "an"):
        return "U"
    def hit(w): return t == w or (t.startswith(w) and not t[len(w)].isalpha())
    if any(hit(w) for w in FW): return "F"
    if any(hit(w) for w in MW): return "M"
    if any(w.startswith(t) for w in FW + MW): return "U"
    return "O"
