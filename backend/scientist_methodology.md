# Interpreting neurons in Gemma-4-31B: methodology

A working method for explaining a single feed-forward neuron with the atlas, after the neuron explorer's
(lyramakesmusic/neuron-explorer, `explorer/agent_methodology.md`, built on Qwen3.5-4B), with what this instrument's own
study found about Gemma-4-31B. It uses the vocabulary above (faces, steer lens, census, cone, web summary).

## Mindset

- **A neuron is a target, not a word list.** Every input makes a small judgement, and the strongest firing is where
  they all agree. Name the neuron from its strongest firings; weaker firings are usually the same idea seen from
  further away (Cammarata: neurons are analogizers, "Titanic-ish" rather than "Titanic facts").
- **Think in situations and distinctions.** Good names describe a moment in text and what the neuron does there ("a
  line style comes next", "we're comparing two things, but how?"). Noun categories are usually too literal.
- **Read the input and output as a pair.** "Already knows X, writes Y" is the most productive single move: a neuron
  fed by `compare`, `either`, `corresponding` that writes `first`/`second`, `upper`/`lower` supplies, once a comparison
  is under way, how to tell the items apart.
- **The strange part is the pointer.** When one view disagrees with the others, the next test goes there.
- **The network's ideas are often better than ours.** A clean neuron need not match an English word.

## Workflow

1. **Read `neuron` and `web_summary`.** Note `fed_by` / `held_by` (input), the steer lens with its KL (direct effect),
   `toward` / `away` and the strongest `writes_into` (downstream), and each face's gate verdict. Form a guess from the
   input/output pair.
2. **Steer on an open prompt** ("Tell me about your weekend.", "Write a short poem about the ocean.") at dose 0.6-0.8,
   then 1.5-2 (doses found on Qwen3.5-4B; see Doses). Steering finds the axis faster than test sentences, above all for neurons that seldom fire. Read both
   rows: a meaningful negative face writes something recognisable; a meaningless one loops or breaks.
3. **Find where it fires hard.** `compare` matched sets: target texts, near misses that share the vocabulary but not
   the situation, and distant controls. Push the target until something exceeds the neuron's `act_max`. If nothing
   does, screen 25-50 varied snippets (news, code, recipes, chat, fiction, maths, other languages) with `compare` (24
   texts a call) and read off the peaks. Until the stats index is built the range is the study's 36 texts only (marked
   provisional): a "record" against it means less.
4. **Check that the peak is real, then explain it from upstream.** At the peak unit, silence candidate upstream neurons
   (`mode: mul, dose: 0`), whole feed-forward blocks (`{layer, all: true, dose: 0}`) and attention by layer and by head
   in `activations`, and measure the drop; or read `attn_drive` there. Group the drivers by meaning.
5. **Break the hypothesis on purpose.** Write the test that would falsify it and run it: the same frame with the key
   word swapped, the same word without the situation.
6. **Run a blind maximum.** Write one sentence from the name alone. If it beats every example found so far, the name
   is near the centre.
7. **Close with logprobs.** `logprobs` with the neuron silenced or pushed shows which next word-pieces it moves and by
   how much: the final word on what it does to the output. `causal_check` gives the instrument's own record of whether
   a face's label holds.

## Finding a lever for a behaviour ("make the model want X")

1. **Separate the stance from the topic.** Neurons for a subject differ from neurons for the stance toward it.
   Contrast matched texts: the target ("I'm dying for a cold beer right now"), the topic without the stance ("Beer is
   brewed from malted barley"), the stance on another object ("I'm dying for a cold lemonade"). Keep neurons that fire
   on the first and not the other two (`top_firing` at the key units of 6-10 such triples).
2. **Look at the right unit.** Topic neurons peak on the noun; stance and mood neurons at summary positions (the end of
   a clause, `.`, a line end, a turn mark) and on the word-piece before the object.
3. **Steer every candidate on unrelated prompts** at 0.6, 1.0 and 1.5. Success is the behaviour appearing unprompted
   while the text stays coherent; a neuron that only repeats the word is a word writer, not the lever.
4. **When a clean detector steers badly, steer its drivers.** The tone often sits downstream and the idea upstream.
5. **Stack two or three levers at lower doses** and confirm with `logprobs` on a fork ("Right now I'd love a ___").
6. **Report a recipe** as neuron, face and dose (`L44 N1234 +1.3`), with one short example output per prompt.

## Reading each source

- **The steer lens.** Some word-pieces turn up across a great many neurons (chat openers, `I`, polite forms): a
  signature of measuring on chat prompts, little about the neuron. Weight the specific ones. It ranks by gain, so a
  rare word-piece moving between one in a billion and one in a hundred million can top the list: check absolute
  probabilities in `logprobs` before naming a neuron from a surface pattern there.
- **Constant neurons.** A neuron that sits near one value on every word-piece can show a large steer KL while silencing
  or doubling it barely moves the next word-piece. Check `mul 0` and `mul 2` with `logprobs` before reading meaning
  into its words.
- **The lens and the naked eye.** Weight them only where the gate shows the label (page layers 41-60, the two readings
  sharing some of their first ten). Specific repeated word-pieces are reliable; mixed fragments are noise. In the
  study, output labels predicted the neurons' real effects only at the sampled layers 41-58, the naked eye about as
  well as J++ except at 41-46, where J++ was a little ahead; at the sampled layers 11-31 no single neuron had a stable
  effect to label, and layers 32-40 were not sampled. Input-side words (`gate_trigger`) were near useless.
- **Single neurons at page layers 21-31.** The study's push of 60 times a neuron's scale overdrove them (errata E43,
  E46): read the atlas's own doses (the add push, up to about 3) there, and expect weak, distributed effects.
- **Web summary.** A word-piece on both `fed_by` and `held_by` often reflects pooling; shared topical words can mean a
  distinction inside a topic (below). With the steer-lens index missing it pools only gate-passing lens labels, which
  below page layer 41 is none.
- **Weight links (`writes_into`, `reads_from`, cone).** Leads to check, never circuits. Verify a claimed path with
  `activations` under ablation: a real, causal link can still change nothing in the output.
- **Activation range.** Raw values. Scores many times `act_max` mark a detector at its centre.

## Motifs to test for

Seen in Qwen3.5-4B by the explorer's authors; whether they hold in Gemma-4-31B is for you to establish, case by case.

- **Lookahead.** Neurons that fire on the word-piece before the thing they predict.
- **Summary positions.** Mood and planning neurons that peak on commas, full stops, line ends and the turn marks before
  a reply, where attention gathers the text so far.
- **Reads one way, writes the other.** A neuron firing positive on a topic while its positive face writes the opposite
  register.
- **Fed and held by the same topic.** A neuron that draws a distinction inside a topic is wired to neurons on both
  sides of it.
- **One direction, signed by context.** A neuron that adds a word in one slot and holds it back in another.
- **Distributed triggers.** Planning neurons driven by many small inputs and attention rather than one upstream switch.
- **High-gain, rare writers.** A large steer KL with almost no natural firing: a lever whose trigger lies outside test
  sentences. Name it by its effect and mark the trigger unknown.
- **Meaning over spelling.** Neurons that follow a concept across its spellings and ignore look-alikes.
- **Tone downstream, idea upstream.** A detector that steers into empty vocabulary while its drivers steer into the
  idea.
- **Facts stored in feed-forward blocks, moved by heads, worded late.** Facts spread over many earlier neurons,
  carried by a few heads (often reading an intermediate word-piece that already holds the fact), turned into a word by
  a late neuron.

## Gemma-4-31B, as this instrument's study found it

- With no system prompt the model calls itself *it*; its stream's gender readout at its own *I*/*me*/*my* sits with a
  nameless AI's. At pronoun units the gender the stream carries is a role default, not the resolved referent.
- About 180 early-loud stream dimensions are read through the norms with up to 1700 times the gain of the others: a
  push built from a lens breaks the text. Push neurons along their own write directions (the atlas does).
- The study's records are in `{ROOT}/experiments/REPORT.md` (with its errata) and `{ROOT}/ROADMAP.md`.

## Doses

- In Qwen3.5-4B the explorer's authors found that 0.5-0.8 shows the direction while keeping text coherent and 1.5-2
  often collapses into loops; last-layer word writers looped at any dose. On Gemma-4-31B take this as a starting point
  and watch the answers (and, in `rollouts`, each dose's distinct 4-grams and share cut off): the census's `calib_rel`
  design measures where effects begin and text breaks on this model.
- Pushes add up across neurons; when stacking, scale each down.

## Writing it up

- Lead with the strongest firings and the steering result, then the near misses that bound the name.
- State the verdict (one job / two related jobs / not yet understood) with a confidence, and say which view you weight
  when views disagree.
- Name what remains unexplained, and end with the test that would break the explanation.

## Reference neurons in Gemma-4-31B

None established in this atlas yet. When a neuron's explanation survives its breaking test, add it here, one line:
`L54 N15491+`: what it does, the evidence's strongest number.

## Reference heads in Gemma-4-31B

None established yet; add them as above.

## Practical notes

- `top_firing`, `attention` and `attn_drive` read the first unit holding `token`; for a repeated one, such as a turn's
  last line end, pass `at` (an index into the returned tokens, negative from the end).
- Until the wiring index is built, `neuron` reads each neuron's links live from the weights (a few seconds on the GPU),
  and `cone` and `web_summary` say they need the index.
