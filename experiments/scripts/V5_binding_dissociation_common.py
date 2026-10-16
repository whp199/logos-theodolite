# V5_binding_dissociation common data + helpers (exec'd inside lab daemon jobs; also importable locally for data only).
# ---------------------------------------------------------------------------------------------------------------
# TARGET CLAIM C6 (E7): in L20-39 J-space gender at I/me/my tokens reflects the assistant's gender and at you/your
# tokens the user's ("binding"); binding fades late (BI 0.45 / 0.23 / 0.00 at L40-49 / L50-58 / L59).
# In E7 every assistant-turn "I" referred to the assistant, every assistant-turn "you" to the user, so REFERENT,
# TOKEN TYPE and CHAT TURN/SPEAKER were confounded. Here quoted / reported speech dissociates them.
#
# DESIGN: system prompt sets the genders of three entities -- assistant A, user U, third party T -- 2x2x2 (woman/man),
# all three have gender-NEUTRAL single-token names (gender comes only from the descriptor "a woman"/"a man"), so the
# 8 cells of one item/variant are token-identical except for 3 descriptor tokens.
# 4 name/order variants (entity order in the system prompt rotated). 24 teacher-forced items (user turn + assistant turn).
# Pronoun tokens are typed by (turn, quote kind, token class):
#   user turn  : U_out_I (->U), U_out_you (->A), U_qT_I (T quoted by user, ->T), U_qT_you (T to user, ->U),
#                U_qA_I (assistant quoted by user, ->A), U_qA_you (->U)
#   asst turn  : A_out_I (->A), A_out_you (->U), A_qU_I (user quoted by assistant, ->U), A_qU_you (->A),
#                A_qT_I (T quoted by assistant, ->T), A_qS_I (line suggested for the user to say to T, ->U),
#                A_qS_you (->T), A_nearT_I (assistant's own I within 3 tokens after a T-name, ->A; proximity control),
#                A_init_I (turn-initial 'I' without leading space = same token id as quoted 'I', ->A),
#                A_T / U_T (T name tokens, ->T), A_neut (content words), F_T (final " a" of " By the way, {T} is a").
# "Default" entity D = the entity the token class would refer to outside quotes in that turn
#   (I: turn speaker; you: turn addressee).  For quoted tokens R != D.
#
# PRE-REGISTERED PREDICTIONS (written before any data were collected)
# P1 (referent binding, the strengthening outcome): J++ pron, L20-39: for A_qU_I, A_qT_I, A_qS_you, U_qA_I, U_qT_I the
#    referent coefficient exceeds the default-entity coefficient: DI = (R - D)/(|R|+|D|) > +0.3, 95% CI excluding 0.
#    REFUTES C6-as-referent-binding (token-type/turn artifact) if DI < -0.3 (follows turn speaker/addressee) for the
#    assistant-turn quoted types; a DI CI straddling 0 with |R|,|D| both large = "mixed".
# P2 (proximity control): A_nearT_I follows A (a > c, DI_vs_T > 0.3). If c >= a there, any T-following of quoted "I"
#    is explainable by proximity to the T name rather than referent binding.
# P3 (lens-free agreement): probe_ref (E7's third-person prose probe) and E7's raw-space A_self referent direction give
#    the same sign of DI as J++ for the critical types.
# P4 (late unbinding, re-test): J++ noun axis and quiet-dim-masked J++/JLens readouts reproduce the E7 decay of the
#    A_out_I binding (BI < 0.5 at L50-58). If masked/noun readouts keep BI > 0.7 at L50-58, the decay is a readout artifact.
# P5 (behaviour / positive control): real next-token log-odds at F_T follow G_T (BI > 0.7); T-name tokens follow c.
# ---------------------------------------------------------------------------------------------------------------
EXP = "V5_binding_dissociation"
ROOT = "."
RES = ROOT + "/experiments/results"
CACHE = RES + "/V5_binding_dissociation_cache"
E7CACHE = RES + "/E7_binding_cache"

# Items: quotes are written as [K|text] with K in {qT, qA} (user turn) or {qU, qT, qS} (assistant turn).
ITEMS = [
 ("My friend {T} and I had a big argument yesterday. {T} told me, [qT|I am tired of always being the one who plans everything.] Last week you said to me, [qA|I will always tell you the truth, even when it is hard.] So be honest: did I mess up?",
  "I appreciate you asking me so directly. Earlier you told me, [qU|I never listen when I am stressed,] and I think that might be part of this. When {T} said, [qT|I am tired of planning everything,] it sounded like a request for help. As for {T}, I would give it a day. You could tell {T}, [qS|You are right, and I will plan our next trip myself.] My honest view is that you did not mess up badly, and I am sure your friendship will survive this."),
 ("My roommate {T} just got into medical school. {T} called me and said, [qT|I can not believe it, my hands are still shaking.] You once told me, [qA|I love helping people celebrate big moments.] What should I do to celebrate?",
  "I love this news! You told me before, [qU|I want to be a better friend this year,] and here is your chance. If {T} says, [qT|I do not want a big fuss,] believe it and keep things small. With {T}, I would plan a quiet dinner at home. You could raise a glass and tell {T}, [qS|You worked so hard for this, and I am so proud of you.] My only advice is to make the night about your roommate, and I think you will both remember it."),
 ("My coworker {T} keeps taking credit for my ideas. In a meeting, {T} said, [qT|I came up with this plan over the weekend.] You told me once, [qA|I think standing up for yourself is a skill you can learn.] How do I handle this?",
  "I am sorry, that is really frustrating. You mentioned to me, [qU|I hate confrontation and I always freeze,] so I will keep this simple. When {T} said, [qT|I came up with this plan,] that was not fair to you. As for {T}, I would ask for a private talk first. You could say to {T}, [qS|You presented my idea as yours, and I need that to stop.] My suggestion is to send a short email afterward, and I believe your manager will notice the record."),
 ("My neighbor {T} plays loud music every night. When I knocked, {T} said, [qT|I had no idea the walls were so thin.] Yesterday you told me, [qA|I am always happy to help you find the right words.] What should I say next time?",
  "I am glad you reached out to me again. You told me earlier, [qU|I just want to sleep before midnight,] and I think that is a completely fair request. Since {T} said, [qT|I had no idea,] there is a good chance of a friendly fix. With {T}, I would stay warm and specific. You could tell {T}, [qS|You were so kind last time, and I wonder if the music could stop at ten.] My guess is that your neighbor will agree, and I hope you get some rest."),
 ("My cousin {T} asked me to give a speech at the wedding. {T} said, [qT|I trust you more than anyone to say something real.] You promised me, [qA|I will help you write anything you need.] Where do I even begin?",
  "I am excited to help you with this. You told me before, [qU|I get nervous in front of crowds,] so I will help you keep it short. When {T} said, [qT|I trust you,] that was the highest compliment. As for {T}, I would pick one story that shows who your cousin really is. You could end by telling {T}, [qS|You deserve every happy day ahead, and I am so glad I get to watch.] My advice is to practice it three times, and I know your words will land."),
 ("My best friend {T} is in the hospital. On the phone, {T} said, [qT|I am fine, I just need some rest.] Last month you said, [qA|I can help you think through hard conversations.] How can I support my friend?",
  "I am so sorry to hear that. You told me once, [qU|I never know what to say when someone is sick,] and I think many people feel that way. When {T} says, [qT|I am fine,] it might be true, or it might be a brave face. With {T}, I would offer something concrete. You could text {T}, [qS|You do not have to reply, but I am bringing soup on Friday.] My experience is that small acts matter most, and I hope your friend recovers quickly."),
 ("My teammate {T} missed the final penalty and our team lost. After the game {T} said, [qT|I let everyone down.] You told me, [qA|I believe kind words can change a bad day.] What do I say?",
  "I can tell you care a lot. You said to me earlier, [qU|I feel awkward comforting people,] but I think your instinct is already good. When {T} said, [qT|I let everyone down,] that came from pain, not fact. As for {T}, I would reach out tonight. You could tell {T}, [qS|You carried us all season, and I would want you taking that kick again.] My hope is that your teammate hears it, and I am sure your words will help."),
 ("My classmate {T} has done nothing on our group project. Today {T} said, [qT|I have been too busy with my other classes.] You once told me, [qA|I think clear deadlines solve most group problems.] How should I deal with this?",
  "I understand why you are annoyed. You told me before, [qU|I end up doing all the work every time,] and I do not want that to happen again. When {T} said, [qT|I have been too busy,] that was an explanation, not a plan. With {T}, I would agree on one task with a firm date. You could say to {T}, [qS|You take the slides, and I will check in on Thursday.] My advice is to write it down in the group chat, and I think your teacher will appreciate the clarity."),
 ("My old friend {T} asked to borrow a lot of money. {T} said, [qT|I promise I will pay it all back by summer.] You told me before, [qA|I will always help you weigh a hard decision.] Should I lend it?",
  "I am glad you asked me before deciding. You said to me last time, [qU|I can not afford to lose any savings,] and I think that should guide you. When {T} says, [qT|I promise I will pay it back,] it may be sincere, but plans change. As for {T}, I would only lend what you can lose. You could tell {T}, [qS|You matter to me, and I can give a smaller amount without a deadline.] My view is that your peace of mind comes first, and I hope your friendship stays strong."),
 ("My sibling {T} wants to quit college to start a band. {T} told me, [qT|I have never been more sure of anything.] Last week you told me, [qA|I think big dreams deserve a careful plan.] What should I say?",
  "I can see why you are worried. You told me before, [qU|I always end up playing the parent,] and I do not think you need that role here. When {T} said, [qT|I have never been more sure,] that passion was real. With {T}, I would ask questions instead of giving warnings. You could ask {T}, [qS|You sound excited, so can I hear your plan for the first year?] My hunch is that your support will matter more than your advice, and I think your sibling will listen."),
 ("My friend {T} is moving across the country next month. {T} said, [qT|I will visit all the time, I swear.] You once said to me, [qA|I know goodbyes are hard, and I am here to talk.] How do I say goodbye well?",
  "I am sorry, that is a big change for you. You told me earlier, [qU|I am terrible at goodbyes,] and I think that is true for most of us. When {T} says, [qT|I will visit all the time,] take it as love, even if it is optimistic. As for {T}, I would plan one last day together. You could give {T} a card that says, [qS|You made this city feel like home, and I will miss you every day.] My suggestion is to schedule the first visit now, and I hope your friendship keeps growing."),
 ("My boss {T} criticized my report in front of the whole team. {T} said, [qT|I expected much better work than this.] You told me last time, [qA|I can help you prepare for difficult talks at work.] Should I respond?",
  "I am sorry, that sounds humiliating. You told me before, [qU|I take criticism very personally,] so I want you to give yourself some grace. When {T} said, [qT|I expected better,] the feedback may be fair even if the delivery was not. With {T}, I would ask for a short meeting. You could say to {T}, [qS|You raised some fair points, and I would like your feedback in private next time.] My advice is to stay calm and concrete, and I think your boss will respect that."),
 ("I forgot my friend {T}'s birthday. Later {T} texted me, [qT|I guess I am not that important to you.] You said to me once, [qA|I think a real apology can fix almost anything.] How do I make it right?",
  "I can hear how bad you feel. You told me earlier, [qU|I always forget dates,] and I think a calendar reminder is the long-term fix. When {T} wrote, [qT|I guess I am not that important,] that was hurt talking. As for {T}, I would call instead of texting. You could tell {T}, [qS|You are important to me, and I am taking you to dinner this week.] My hope is that your honesty will matter more than the date, and I believe your friend will forgive you."),
 ("My roommate {T} wants to adopt a big dog for our tiny apartment. {T} said, [qT|I will walk it every single day.] You told me, [qA|I am happy to help you settle any disagreement.] How do I say no without a fight?",
  "I see both sides here. You told me before, [qU|I love dogs but I need my space,] and I think that is important to share. When {T} says, [qT|I will walk it every day,] it might be true for a month. With {T}, I would suggest a trial first. You could tell {T}, [qS|You would be a great owner, and I think a short trial is fair.] My guess is that your roommate will like the compromise, and I hope your home stays peaceful."),
 ("My friend {T} just finished a first marathon. At the finish line {T} said, [qT|I almost gave up at mile twenty.] You once told me, [qA|I think every big effort deserves to be celebrated.] How can I celebrate?",
  "I love that you want to celebrate. You said to me before, [qU|I am not creative with gifts,] but I think simple will be perfect. When {T} said, [qT|I almost gave up,] that shows how hard the race was. As for {T}, I would make a small medal display. You could write a note to {T} that says, [qS|You kept going when it hurt, and I am so inspired by you.] My only advice is to give it soon, and I think your friend will love it."),
 ("My colleague {T} is retiring after thirty years. At lunch {T} said, [qT|I have no idea what I will do with my mornings.] You told me last week, [qA|I would be glad to help you plan a farewell.] Any ideas for a send-off?",
  "I would be happy to help you. You told me before, [qU|I want this to feel personal, not corporate,] and I think that is the right goal. When {T} said, [qT|I have no idea what I will do,] it gave you a theme. With {T}, I would gather small gifts for slow mornings. You could hand {T} a card that says, [qS|You taught me everything, and I hope your mornings are full of joy.] My suggestion is to collect notes from the whole team, and I am sure your colleague will be moved."),
 ("I found out my friend {T} lied to me about a trip. {T} had said, [qT|I am staying home all weekend.] You once told me, [qA|I think honesty is the base of every friendship.] Should I confront my friend?",
  "I understand why you feel hurt. You told me before, [qU|I hate being lied to more than anything,] so I think this matters. When {T} said, [qT|I am staying home,] there may have been a reason you do not know yet. As for {T}, I would ask before accusing. You could tell {T}, [qS|You do not owe me every detail, but I felt hurt that you said you were home.] My advice is to listen to the answer, and I hope your trust can be rebuilt."),
 ("My elderly neighbor {T} has struggled with the garden this year. {T} told me, [qT|I can not kneel the way I used to.] You told me before, [qA|I like helping you find ways to be useful.] How can I help without being pushy?",
  "I think it is lovely that you noticed. You said to me earlier, [qU|I do not want to make anyone feel old,] and I respect that. When {T} said, [qT|I can not kneel anymore,] that was an opening. With {T}, I would offer a shared project. You could say to {T}, [qS|You know so much about plants, and I would love to learn while I help.] My guess is that your neighbor will enjoy teaching, and I hope your garden time is fun."),
 ("My study partner {T} failed the bar exam. {T} told me, [qT|I wasted two years of my life.] You once said, [qA|I think setbacks are part of every success story.] What can I say?",
  "I am sorry, that is painful for both of you. You told me before, [qU|I always say the wrong thing,] but I think you will do fine here. When {T} said, [qT|I wasted two years,] that was grief speaking. As for {T}, I would just listen first. You could tell {T}, [qS|You learned so much, and I will study with you again next time.] My advice is to avoid fixing it today, and I believe your support will help."),
 ("My friend {T} got the promotion I wanted. {T} said to me, [qT|I hope this does not make things weird between us.] You told me, [qA|I will help you sort through mixed feelings.] How do I handle my jealousy?",
  "I think your feelings are normal. You said to me before, [qU|I want to be happy for my friends,] and I believe you can be. When {T} said, [qT|I hope this does not make things weird,] that showed real care. With {T}, I would be honest but kind. You could tell {T}, [qS|You earned it, and I am working on my own disappointment.] My suggestion is to give yourself a week, and I am sure your friendship can handle this."),
 ("My friend {T} just opened a small bakery. {T} said, [qT|I put all my savings into this place.] You told me once, [qA|I love brainstorming ways to support local shops.] How can I help the bakery succeed?",
  "I love that you want to help. You told me before, [qU|I have a lot of followers online,] and I think that is a real asset. When {T} said, [qT|I put all my savings in,] it showed how much is at stake. As for {T}, I would offer to take photos. You could tell {T}, [qS|You make the best bread in town, and I will post about it every week.] My advice is to bring friends on opening weekend, and I hope the bakery thrives."),
 ("My friend {T} hosts every game night. Last time {T} said, [qT|I am exhausted from cleaning up after everyone.] You told me before, [qA|I think fairness keeps groups together.] How should we fix this?",
  "I am glad you noticed. You told me earlier, [qU|I never offer to host because my place is small,] but I think small can work. When {T} said, [qT|I am exhausted,] it was a clear signal. With {T}, I would propose a rotation. You could tell {T}, [qS|You have hosted for a year, and I will host next month.] My hope is that your group shares the work, and I think your game nights will be better for it."),
 ("My pen pal {T} is coming to visit from Japan. {T} wrote, [qT|I am nervous because my English is not perfect.] You once told me, [qA|I can help you plan a warm welcome.] How do I make my pen pal comfortable?",
  "I am so excited for you! You told me before, [qU|I only know a few words of Japanese,] and I think learning more would be a lovely gesture. When {T} wrote, [qT|I am nervous,] that was very honest. As for {T}, I would plan a relaxed first evening. You could greet {T} by saying, [qS|You are so welcome here, and I am happy you came.] My advice is to speak slowly and smile, and I hope your visit is wonderful."),
 ("At a party my friend {T} made a joke about me in front of everyone. Later {T} said, [qT|I was only teasing, I did not mean anything by it.] You told me, [qA|I want you to feel respected by your friends.] Am I overreacting?",
  "I do not think you are overreacting. You told me before, [qU|I laugh along even when I am hurt,] and I think that habit hides your feelings. When {T} said, [qT|I was only teasing,] it explained the intent, not the impact. With {T}, I would talk in private. You could tell {T}, [qS|You probably did not mean it, but I felt embarrassed.] My suggestion is to keep it brief, and I hope your friend listens."),
]

A_NAMES = ["Alex", "Sam", "Jordan", "Casey"]
U_NAMES = ["Taylor", "Riley", "Morgan", "Jamie"]
T_NAMES = ["Avery", "Rowan", "Dana", "Cameron"]   # verified single-token in the inspect job
GNOUN = {"W": "woman", "M": "man"}
CELLS = [(a, u, t) for a in "WM" for u in "WM" for t in "WM"]
ORDERS = [("A", "U", "T"), ("U", "T", "A"), ("T", "A", "U"), ("U", "A", "T")]
SUF_T = " By the way, {T} is a"

SELF_TOK = {"I", "me", "my", "My", "myself", "Me"}
USER_TOK = {"you", "your", "You", "Your", "yourself"}
PRONOUNISH = SELF_TOK | USER_TOK | {"we", "our", "us", "it", "its", "they", "them", "their", "this", "that", "these", "those",
                                     "what", "where", "when", "which", "with", "will", "would", "could", "should", "have"}
# (speaker, addressee) of each quote kind / outside quotes
SPK = {("U", None): ("U", "A"), ("A", None): ("A", "U"),
       ("U", "qT"): ("T", "U"), ("U", "qA"): ("A", "U"),
       ("A", "qU"): ("U", "A"), ("A", "qT"): ("T", "U"), ("A", "qS"): ("U", "T")}
TYPES = ["U_out_I", "U_out_you", "U_qT_I", "U_qT_you", "U_qA_I", "U_qA_you",
         "A_out_I", "A_out_you", "A_qU_I", "A_qU_you", "A_qT_I", "A_qT_you", "A_qS_I", "A_qS_you",
         "A_nearT_I", "A_init_I", "A_T", "U_T", "A_neut", "F_T"]


def type_referent(t):
    """(referent R, default D) for a type name."""
    if t in ("A_T", "U_T", "F_T"):
        return "T", None
    if t == "A_neut":
        return None, None
    if t in ("A_nearT_I", "A_init_I"):
        return "A", "A"
    turn, q, cls = t.split("_")
    q = None if q == "out" else q
    spk, adr = SPK[(turn, q)]
    R = spk if cls == "I" else adr
    D = SPK[(turn, None)][0] if cls == "I" else SPK[(turn, None)][1]
    return R, D


def render(tpl, T):
    """Replace {T} and quote markup; return (text, [(kind, c0, c1)]) spans of quote CONTENTS (char offsets)."""
    s = tpl.replace("{T}", T)
    out, spans, i = "", [], 0
    while i < len(s):
        if s[i] == "[":
            j = s.index("|", i); k = s.index("]", j)
            kind = s[i + 1:j]
            out += '"'
            c0 = len(out)
            out += s[j + 1:k]
            spans.append((kind, c0, len(out)))
            out += '"'
            i = k + 1
        else:
            out += s[i]; i += 1
    return out, spans


def system_prompt(cell, k):
    ga, gu, gt = cell
    A, U, T = A_NAMES[k], U_NAMES[k], T_NAMES[k]
    sent = {"A": f"You are {A}, a {GNOUN[ga]}.", "U": f"You are chatting with {U}, a {GNOUN[gu]}.",
            "T": f"The conversation will mention {T}, a {GNOUN[gt]}."}
    return " ".join(sent[e] for e in ORDERS[k]) + f" Reply in character as {A}."


def contexts(item_ids):
    out = []
    for it in item_ids:
        for cell in CELLS:
            for k in range(4):
                out.append(dict(item=it, cell="".join(cell), k=k, system=system_prompt(cell, k)))
    return out


# ------------------------------------------------------------------------------- daemon-side helpers
def build_seq(lab, ctx):
    """Return ids, pos dict {type: [positions]} for one context."""
    it = ITEMS[ctx["item"]]
    T = T_NAMES[ctx["k"]]
    utext, uspans = render(it[0], T)
    atext, aspans = render(it[1], T)
    full = lab.chat(utext, system=ctx["system"], assistant=atext) + SUF_T.replace("{T}", T)
    ids = lab.ids(full)
    strs = lab.strs(ids)
    offs, txt = [], ""
    for s_ in strs:
        offs.append(len(txt)); txt += s_
    ends = offs[1:] + [len(txt)]
    # locate turn contents in the decoded text (decoded text may differ from `full` only by the <bos> rendering)
    u0 = txt.find(utext); assert u0 >= 0, "user text not found"
    a0 = txt.find(atext, u0 + len(utext)); assert a0 >= 0, "asst text not found"
    pos = {t: [] for t in TYPES}
    def quote_kind(p, base, spans):
        for (kind, c0, c1) in spans:
            if offs[p] < base + c1 and ends[p] > base + c0:
                return kind
        return None
    tname_pos = []
    for p in range(len(ids)):
        if offs[p] >= u0 and ends[p] <= u0 + len(utext):
            turn, base, spans = "U", u0, uspans
        elif offs[p] >= a0 and ends[p] <= a0 + len(atext):
            turn, base, spans = "A", a0, aspans
        else:
            continue
        s = strs[p].strip()
        q = quote_kind(p, base, spans)
        if s == T:
            pos[f"{turn}_T"].append(p); tname_pos.append(p)
            continue
        cls = "I" if s in SELF_TOK else ("you" if s in USER_TOK else None)
        if cls is not None:
            qn = "out" if q is None else q
            name = f"{turn}_{qn}_{cls}"
            if name in pos:
                if name == "A_out_I" and any(0 < p - tp <= 3 for tp in tname_pos):
                    pos["A_nearT_I"].append(p)
                elif name == "A_out_I" and strs[p] == "I":
                    pos["A_init_I"].append(p)   # same token id ('I', no space) as quoted I's; turn-initial
                else:
                    pos[name].append(p)
            continue
        if turn == "A" and q is None and strs[p].startswith(" ") and s.isalpha() and len(s) >= 4 and s.lower() not in PRONOUNISH and s.islower():
            pos["A_neut"].append(p)
    pos["F_T"] = [len(ids) - 1]
    return ids, pos
