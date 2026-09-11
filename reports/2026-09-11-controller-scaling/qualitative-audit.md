# Qualitative audit of the controller scaling pilot

This audit reads the frozen pilot questions, normalized corpus, original MuSiQue
source paragraphs and completed generation traces. It distinguishes missing
relations in benchmark annotations from failures to use sufficient delivered
evidence. These examples are descriptive, selected after inspecting outcomes;
they neither change the twelve-question cohort nor replace its recorded scores.

Source references below identify JSONL records by `id`, not mutable line numbers.
Questions are in `data/controller-scaling/dev.jsonl`; corpus records are in
`data/controller-scaling/corpus.jsonl`, with document prose in `text`.
The original question/decomposition/paragraph records are preserved in
`data/controller-scaling/provenance/musique_ans_v1.0_dev.jsonl.gz`.
The [data protocol](../../docs/controller-scaling-data.md) records hashes and provenance.
Run observations refer to `runs/controller-scaling/pilot-v1/JOB/results.jsonl`,
selected by `example_id`, with controller actions in
`trace[event=decision].raw_action` and scored final responses in `prediction`.

## Annotation exposure does not guarantee sufficient evidence

**Smokey Bear's voice actor and spouse — `2hop__84172_198548`.**
The gold answer is Katharine Ross. The source's spouse decomposition
(`question_decomposition[id=198548]`) points to paragraph 16, *Murder in Texas*.
That paragraph says the film stars “Katharine Ross, Sam Elliott, Farrah Fawcett,
and Andy Griffith.” It does not state who married whom. The other annotated
paragraph identifies several Smokey Bear voice actors, including Sam Elliott.

- *Murder in Texas*: `musique-paragraph-33b9effbd7f204b966ab57bc2593c95e75712f5f8bd23f58e70a3d384bca9db9`.
- *Smokey Bear*: `musique-paragraph-44846b59a0d5e0cadb874985ad4ff51c82d721f8be2d1eac9bf9a62000fb7934`.

The `14b_thinking` run searches “Spouse of Sam Elliott” and delivers both
annotated paragraphs without truncation, then returns UNKNOWN. The annotated
paragraphs alone do not establish the requested marriage, so this example
cannot establish a reasoning failure merely from its 100% support-ID coverage.
The missing relation exists in the original benchmark source; normalization did
not remove it. This observation is not a claim that no other corpus paragraph
could establish the relation.

**Paul Kane's residence at death — `3hop1__210639_147339_47686`.**
The residence decomposition (`question_decomposition[id=147339]`) points to
paragraph 3, *Paul Kane*, normalized as
`musique-paragraph-a4f6302d5db85f1a6d3bbf391ae7fd398ba0c8826f57a806f74fd5a88a0626ed`.
It says he “grew up in Toronto” and describes journeys departing from Toronto.
It does not say where he lived when he died. That missing temporal relation
matters to the question's chain, even though Toronto is the decomposition answer.
The remaining annotations identify his painting and the Toronto Coach Terminal;
they do not repair the residence-at-death assertion in this cited paragraph.

**Markus Zusak's citizenship and a neighboring country —
`4hop1__58323_375563_161848_53331`.**
The source assigns the citizenship hop (`id=375563`, paragraph 6) to *The
Messenger (Zusak novel)*:
`musique-paragraph-d3534acad149e64279f7c78c25b4eaafb79e949ec6e771dfe3dbb341ab97dc7c`.
Its complete sentence identifies Zusak as author and names an Australian book
award. It does not state his citizenship. The neighboring-country hop
(`id=161848`, paragraph 3) points to *1952 Winter Olympics*:
`musique-paragraph-eb1e5f2b94d1652f13647cdfc59ee041c11bb07ae431ad6484ba22611b952488`.
It states “New Zealand and Portugal took part ... for the first time” and says
Australia returned after an absence. It does not establish geographic proximity.
These labels expose co-occurring entities, without themselves establishing all
the relations required by the composed question.

## Useful discovery can be lost during final synthesis

**Vatican City — `3hop1__64957_87694_64412`.**
In `8b_short`, only one of three annotated supports is delivered. In
`8b_thinking`, subsequent searches deliver all three:

- *The Last Supper (Leonardo)*: `musique-paragraph-f46710cf989e1a4adc3fb9be1319d786f0aa6d9e4b4134763984958f1955741a` identifies Peter holding the knife.
- *St. Peter's Basilica*: `musique-paragraph-7f84b41792d166d84051cd10f610930c7966d0a23effcaf68a9c3e82b9a7afb7` locates the basilica in Vatican City.
- *Vatican City*: `musique-paragraph-dbe3a9f7245186633613d25a4bd04a683e25b3884e144552910d8c0bbf8e8dd6` gives “11 February 1929” and states the treaty established the modern city-state.

These three supports are not in the run's truncated-document list. The thinking
controller eventually emits `ANSWER` followed by `1929`; that is a partial date,
not the exact gold answer. The separate scored final generation returns UNKNOWN.
This trajectory demonstrates additional relevant retrieval, without a final
accuracy gain.

**Papa Roach, Veoh and San Diego — `4hop3__312119_132409_371500_35031`.**
The short 8B controller exposes three of four annotated supports; its thinking
counterpart exposes all four, with no delivered documents truncated:

- *Getting Away with Murder (song)*: `musique-paragraph-536c08fbd05e01035fd39d83cbd375b2aa03f719301463091a4dfdec7fd65873` identifies Papa Roach as performer.
- *Papa Roach*: `musique-paragraph-846d359865d5969e63b147d241cfdbddd2e5cbf31fa8db9f2de9021d3bf54a79` places the band's formation in California.
- *Veoh*: `musique-paragraph-2bf015804b80b2a587a372caf630c880fc65d395bb42138106315c1f51e50f7f` locates the company in San Diego.
- *San Diego*: `musique-paragraph-7bec89a271a602679c3f70bb79458f16dbd36c053ff9954887c68276902ba4a9` calls its urban area “third-largest ... in the state.”

The `8b_thinking` controller's last action explicitly states the correct rank:
“ANSWER: San Diego is the third-largest urban area in California”. The frozen
protocol treats this as a finish signal, discards its payload, and invokes the
separate final generator. That call outputs `SEARCH: What is the state where
Papa Roach was formed?`, which is a search command rather than an answer.
The scored failure therefore coexists with useful discovery and a correct
controller answer statement. It does not demonstrate that the controller's
reasoning would be retained in a latent-memory architecture.

**Cairo University's ranking — `2hop__403580_37168`.**
Both annotated paragraphs are initially available without truncation:
*Said Ashour*, `musique-paragraph-87bac31e983b50141347a3712ed7b084d392cfe046b098ff3a365a9975507a46`,
and *Egypt*, `musique-paragraph-a4d135d08c9bc82f11dbdbefe3c18065427cf72dbe5f851b5a2dfd544986e3e4`.
The latter supplies Cairo University's QS rank as `551-600`. The 8B thinking
trace explicitly notices that rank but hesitates because Ashour worked at
several institutions. Its final answer is UNKNOWN. The short 8B final names
Cairo University rather than providing a ranking. This is a synthesis/task
interpretation issue on available evidence, with genuine ambiguity about which
employer the question intends.

## Interpretation

Supporting-ID counts are reproducible annotation-exposure measures, not a
semantic guarantee that the answer follows from the retrieved text. Conversely,
zero final exact match can hide useful intermediate search progress. Final
responses beginning with SEARCH must remain wrong answers; incidental token-F1
overlap in such commands is not successful answering.

The protocol keeps controller reasoning transient and the final answer
nonthinking. Its mixed system prompt exposes both search and answer commands to
that final call. The observations above motivate examining final synthesis and
instruction following separately; they do not prove which prompt change would
fix the failures. A replay on preserved evidence would be a separate diagnostic,
with original outputs and end-to-end pilot measurements retained.
