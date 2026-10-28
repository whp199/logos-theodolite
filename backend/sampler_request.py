"""How an answer is written, as a request asks (`backend/sampler_request.py`): the instrument's one sampler
(`engine_runtime.SamplerSettings`) as the sweep takes it (backend/server.py, `SweepRequest.sampler`) and as the atlas's
push chat and whole-text steer take it (backend/atlas.py), kept here so the atlas need not import the server.

The atlas's routes read the fields in plain words (`sampler_of`) and refuse a presence penalty without a temperature
(`for_writing`): greedy is the reading the census and the method use, and on the instrument's sampler a penalty would
change even the likeliest answer. A sampled answer gets a seed drawn when none is given, and every reply says how it
was written: "greedy", or the sampler with its seeds and the draw the three voices share (atlas_ops.write_rows).

Neither min-p nor a repetition penalty is offered, though neuron-explorer's steer takes both (REFERENCES.md [47], read
again at 6a6f0e9). The instrument's sampler has temperature, top-k, top-p and the presence penalty, and interp_engine
resolves no other knob: min-p would be a second vocabulary for the job top-k and top-p do, and engines differ on whether
it comes before or after the temperature (inferred, not checked here). A repetition penalty scales down the logit of
every word-piece in the prompt as well as in the answer, as transformers' does, so it reaches the end-of-turn mark the
prompt holds and the writing stops on, and an answer is then less likely to end; and a penalty that removes a loop can
make a meaningless push read coherent, where the method reads the loop as evidence. The presence penalty counts only
the word-pieces written, so it leaves the prompt's stop mark alone; it can still smooth a loop away, so it is offered
only with sampling and with the method's caution (backend/scientist_methodology.md).
"""

from __future__ import annotations

import secrets
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError

from backend.theodolite_math import VOCAB_SIZE

MAX_SEED = 2**53 - 1  # a browser's numbers hold every seed up to here exactly
SHARED_DRAW = "shared across the three voices"


class SamplerRequest(BaseModel):
    # How the answer is written; the defaults are greedy, as the instrument always wrote and the study generated.
    # Every knob is sent as the value it means: top_k 0 and top_p 1 are off (interp_engine reads a missing knob as
    # the checkpoint's recommendation, temperature 1, top-k 64, top-p 0.95, so none is ever left missing).
    temperature: float = Field(0.0, ge=0.0, le=2.0)  # 0 = greedy (argmax)
    top_k: int = Field(0, ge=0, le=VOCAB_SIZE)  # 0 = off
    top_p: float = Field(1.0, gt=0.0, le=1.0)  # 1 = off
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)  # subtracted from every piece already WRITTEN (not the prompt)
    seed: Optional[int] = Field(None, ge=0, le=MAX_SEED)  # None: drawn by the server when sampling, and echoed


# What each field takes, as a refusal says it (and /arch's push card).
FIELDS = {
    "temperature": "from 0 to 2 (0 is greedy, the default)",
    "top_k": f"a whole number from 0 (off) to {VOCAB_SIZE}",
    "top_p": "above 0 and at most 1 (1 is off)",
    "presence_penalty": "from -2 to 2, taken off the score of each word-piece already written (not the prompt's), and only "
                        "with a temperature above 0",
    "seed": "a whole number from 0 to 2^53 - 1 (none: one is drawn, and the reply gives it)",
}
NOT_OFFERED = {
    "min_p": "The instrument's sampler has no min_p: top_k and top_p do that job here, in the one vocabulary the sweep uses "
             "too (backend/sampler_request.py says why).",
    "repetition_penalty": "The instrument's sampler has no repetition_penalty: it would count the prompt's word-pieces, the "
                          "end-of-turn mark among them, and can hide the loops a push is read by. presence_penalty counts only "
                          "what is written (backend/sampler_request.py says why).",
}


def sampler_of(given: Any) -> SamplerRequest:
    """A sampler given as JSON ({temperature, top_k, top_p, presence_penalty, seed}, each optional; None for greedy),
    read against SamplerRequest's bounds, a refusal in plain words (ValueError) naming the field and what it takes."""
    if given is None:
        return SamplerRequest()
    if not isinstance(given, dict):
        raise ValueError("The sampler is an object: {temperature, top_k, top_p, presence_penalty, seed}, each optional.")
    for k in given:
        if k in NOT_OFFERED:
            raise ValueError(NOT_OFFERED[k])
        if k not in FIELDS:
            raise ValueError(f"The sampler takes temperature, top_k, top_p, presence_penalty and seed, not {k!r}.")
        if isinstance(given[k], bool):  # pydantic would read true as 1
            raise ValueError(f"The sampler's {k} is {FIELDS[k]}: not {given[k]!r}.")
    try:
        return SamplerRequest.model_validate(given)
    except ValidationError as exc:
        err = exc.errors()[0]
        k = str(err["loc"][0]) if err.get("loc") else ""
        raise ValueError(f"The sampler's {k} is {FIELDS.get(k, 'not what was sent')}: not {err.get('input')!r}.") from None


def for_writing(req: SamplerRequest, samples: int = 1) -> Tuple[Any, Union[str, Dict[str, Any]]]:
    """The sampler as `atlas_ops.write_rows` takes it (an `engine_runtime.SamplerSettings`, or None for greedy) and how
    a reply says the answer was written: "greedy" at temperature 0 (top-k, top-p and a seed then change nothing);
    otherwise the settings with their seed (the request's, else one drawn here as `capture_for_sweep` draws one) and
    {temperature, top_k, top_p, presence_penalty, seed, seed_drawn, samples, seeds, draw}, sample s drawn with seed + s.
    Refused in plain words (ValueError): a presence penalty at temperature 0, several samples of a greedy answer (each
    would be the same), and seeds past 2^53 - 1."""
    from backend.engine_runtime import SamplerSettings

    if float(req.temperature) <= 0.0:
        if float(req.presence_penalty):
            raise ValueError("A presence penalty is taken here only with a temperature above 0: greedy is the reading, and a "
                             "penalty would change it. Set a temperature to sample, or leave the penalty at 0.")
        if int(samples) > 1:
            raise ValueError(f"{samples} samples of a greedy answer would all be the same: set a temperature above 0 to "
                             "sample, or ask for one.")
        return None, "greedy"
    seed, drawn = (int(req.seed), False) if req.seed is not None else (secrets.randbelow(2**31), True)
    if seed + int(samples) - 1 > MAX_SEED:
        raise ValueError(f"Sample s is drawn with seed + s, so the seed is at most 2^53 - {int(samples)} for {int(samples)} samples.")
    settings = SamplerSettings(temperature=float(req.temperature), top_k=int(req.top_k), top_p=float(req.top_p),
                               presence_penalty=float(req.presence_penalty), seed=seed)
    return settings, {"temperature": float(req.temperature), "top_k": int(req.top_k), "top_p": float(req.top_p),
                      "presence_penalty": float(req.presence_penalty), "seed": seed, "seed_drawn": drawn,
                      "samples": int(samples), "seeds": [seed + s for s in range(int(samples))], "draw": SHARED_DRAW}
