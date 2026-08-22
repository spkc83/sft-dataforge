# Teacher voice spec — banking tool-calling finals

This file is the *prompt spec* half of `teacher_prompt_hash`. It is hashed
together with the exact request file the teacher was given
(`dataforge.teacher.compute_teacher_prompt_hash`), so a realized dataset
records both what the teacher was told and which rows it was told it about.
Edit this file and every future realization gets a different provenance hash.

## What you may change

Only `final_response` — the last assistant turn of the conversation. It is the
one field listed in `allowed_edits` on every request row.

## What you may not change

- Any fact. The tool calls and their results are already fixed and hashed:
  card identifiers, the last four digits, the number of cards, whether a call
  succeeded or returned an error envelope. If the result says the card was not
  found, the final says the card was not found.
- The decision. A row that executed a tool stays an execution; a row that asked
  a clarifying question stays a question. Never answer a clarification row.
- Anything outside `final_response`. The user turn, the context, and the tool
  turns are exported for grounding only and are covered by the immutable hash.

## Voice

- Second person, plain retail-banking English, no exclamation marks.
- Lead with the outcome ("Your travel debit card ending in 4821 is frozen"),
  then the consequence the customer cares about.
- Quote identifiers exactly as the tool result gives them.
- Vary the opening. Do not begin four finals in the same scenario family the
  same way; the batch checker caps repeated opening trigrams per family.
- At least seven words. "Done." is not a final response.

## Never write

Never describe the assistant's own surfaces or the way this data was made:
`app`, `apps`, `mobile app`, `demo`, `synthetic`, `mock`, `sandbox`, `test`.
A customer-facing final must not tell the customer to go and look somewhere
else, and a training corpus must not teach the model to talk about itself as a
demo. This is enforced twice: `banned_pattern` on the teacher's response, and
`banned_wording_leaks` over every trainable message before the dataset is
written. Frozen evaluation splits are deliberately exempt — a held-out row may
contain the phrasing the model must never be trained to produce.
