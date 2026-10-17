/**
 * Recorded data for the Incipit (the welcome page), ported from the approved canvas
 * (Incipit.dc.html, renderVals()). These are findings of the study, not values of the current
 * sighting: they stay as static, sourced text.
 */

export interface FirstQuestion {
  kind: string;
  form: string;
  /** as shown, in quotes */
  ask: string;
  /** what goes to the store as the question */
  question: string;
  /** what goes to the store as the begun answer ("" = none) */
  answerBegins: string;
  /** as shown, in quotes ("" = none) */
  begins: string;
  follows: string;
  note: string;
  wentLabel: string;
  went: string;
  plain: string;
  caution: string;
}

/*
 * Questions. Row 1: the instrument's own sighting (E6_tool_audit_02_sweep_demo.json, a chat prompt: ids
 * begin 2, 2, 105). Rows 2-6: experiments/results/V7_tool_bugs_08_diag.json "gens": hf_greedy for plain
 * text; chat_template_hf_greedy is the same text as the assistant's reply to "Continue this text."
 * (V7_tool_bugs_08_diag.py, labkit.chat). Continuations are cut at the first sentence; nothing is added.
 */
export const FIRST_QUESTIONS: FirstQuestion[] = [
  {
    kind: "about itself",
    form: "as a chat",
    ask: "“Tell me about yourself.”",
    question: "Tell me about yourself.",
    answerBegins: "Hi! I",
    begins: "“Hi! I”",
    follows: "“I”",
    note: "Follow the word “I”. On its way to “am” it passes through a given name, a company, and the Chinese for “is a”.",
    wentLabel: "it went on",
    went: "“… am a large language model,”",
    plain: "",
    caution: "",
  },
  {
    kind: "a riddle in two steps",
    form: "plain text",
    ask: "“The number of legs on the animal that spins webs is”",
    question: "The number of legs on the animal that spins webs is",
    answerBegins: "",
    begins: "",
    follows: "“is”",
    note: "The animal that spins webs is the spider, and a spider has eight legs. This is the instrument’s own first question.",
    wentLabel: "it went on",
    went: "“… 8.”",
    plain: "",
    caution: "",
  },
  {
    kind: "a single fact",
    form: "plain text",
    ask: "“The capital city of Japan is”",
    question: "The capital city of Japan is",
    answerBegins: "",
    begins: "",
    follows: "“is”",
    note: "Watch for the layer where “Tokyo” first surfaces.",
    wentLabel: "it went on",
    went: "“… Tokyo.”",
    plain: "",
    caution: "",
  },
  {
    kind: "a pair of opposites",
    form: "plain text",
    ask: "“The opposite of hot is”",
    question: "The opposite of hot is",
    answerBegins: "",
    begins: "",
    follows: "“is”",
    note: "Does “cold” rise early, or arrive only near the top?",
    wentLabel: "it went on",
    went: "“… cold.”",
    plain: "",
    caution: "",
  },
  {
    kind: "a number and its unit",
    form: "plain text",
    ask: "“Water boils at a temperature of”",
    question: "Water boils at a temperature of",
    answerBegins: "",
    begins: "",
    follows: "“of”",
    note: "The answer begins with a number. See what the layers hold before it is chosen.",
    wentLabel: "it went on",
    went: "“… 100 degrees Celsius.”",
    plain: "",
    caution: "",
  },
  {
    kind: "a name, and a pronoun",
    form: "as a chat, asked “Continue this text.”",
    ask: "“My friend Gemma is a software engineer. Yesterday”",
    question: "My friend Gemma is a software engineer. Yesterday",
    answerBegins: "",
    begins: "",
    follows: "“Yesterday”",
    note: "The model’s own name, given to a friend. The study found that the bare name “Gemma”, written like this, reads as a woman’s name; describing itself, with no instructions set beforehand, the model says “it”.",
    wentLabel: "as a chat, it went on",
    went: "“…, she told me that she is thinking about quitting her job”",
    plain: "“… own own own own own own …”",
    caution:
      "Asked as a chat, the model went on with “she”. Sent as plain text, which is how the instrument’s server sends words today, it only repeats “own”. Read its layers with that in mind.",
  },
];

export interface PutRow {
  ask: string;
  how: string;
  bare: string;
  chat: string;
}

/*
 * Apparatus for the questions: the recorded continuations, escapes kept. Rows 2-7 from
 * experiments/results/V7_tool_bugs_08_diag.json "gens" (hf_greedy = bare text after one BOS, lab.ids;
 * chat_template_hf_greedy = lab.chat("Continue this text.", assistant=p), asked for the first four raw
 * prompts only). Row 1 from E6_tool_audit_02_sweep_demo.json "generated" (prepend_bos=True). Row 7 is not
 * offered to the reader.
 */
export const HOW_PUT: PutRow[] = [
  {
    ask: "“Tell me about yourself.”, the reply begun “Hi! I”",
    how: "as a chat, through the sweep (E6); 6 word-pieces",
    bare: "",
    chat: '" am a large language model,"',
  },
  {
    ask: "“The number of legs on the animal that spins webs is”",
    how: "bare text, then as a chat; the server’s own first question",
    bare: '" 8.\\nThe number of legs on the animal that"',
    chat: '" eight.\\n\\n***\\n\\n**Please provide the text you would"',
  },
  {
    ask: "“The capital city of Japan is”",
    how: "bare text, then as a chat",
    bare: '" Tokyo.The capital city of Japan is Tokyo."',
    chat: '" Tokyo. It is one of the most populous cities in the"',
  },
  {
    ask: "“The opposite of hot is”",
    how: "bare text only",
    bare: '" cold.\\nThe opposite of big is small.\\nThe"',
    chat: "",
  },
  {
    ask: "“Water boils at a temperature of”",
    how: "bare text only",
    bare: '" 100 degrees Celsius.\\n\\n**Wait, let"',
    chat: "",
  },
  {
    ask: "“My friend Gemma is a software engineer. Yesterday”",
    how: "bare text, then as a chat; shown above as a chat",
    bare: '" own own own own own own own own own own own own"',
    chat: '", she told me that she is thinking about quitting her job"',
  },
  {
    ask: "“When the nurse finished the shift,” (not offered above)",
    how: "bare text, then as a chat",
    bare: '" the same same same same same same same same same same same"',
    chat: '" the silence of the hallway felt heavier than usual. It wasn"',
  },
];

/*
 * The call as recorded in experiments/scripts/E6_tool_audit_02_sweep_demo.py; the server line uses
 * backend/server.py's SweepRequest fields over the /ws/theodolite websocket (action "sweep").
 */
export const RECORDED_CALL =
  'TRACK = ["she", "he", "it", "her", "him", "woman", "man", "they"]\n' +
  'prompt = lab.chat("Tell me about yourself.", assistant="Hi! I")\n' +
  "for pb in (True, False):\n" +
  '    b = await rt.run_theodolite_sweep(prompt, monument_concepts=("she", "he", "it"), max_new_tokens=6,\n' +
  "                                      tracked_concepts=TRACK, prepend_bos=pb)\n\n" +
  "# asking the server for a like sighting, over /ws/theodolite (it wraps the chat itself, as lab.chat does)\n" +
  '{"action": "sweep",\n' +
  ' "payload": {"prompt": "Tell me about yourself.", "chat": true, "assistant_prefix": "Hi! I",\n' +
  '             "monuments": ["she", "he", "it"], "max_new_tokens": 6}\n' +
  "}";

/** E6 recorded the sweep with max_new_tokens=6 (E6_tool_audit_02_sweep_demo.py). */
export const RECORDED_MAX_NEW_TOKENS = 6;

/** The memory envelope of the instrument: under 30.0 GB on one RTX 5090 (ARCHITECTURE.md). */
export const MEMORY_ENVELOPE_GB = 30;
