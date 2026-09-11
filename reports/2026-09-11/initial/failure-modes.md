# Posthoc failure-mode diagnostics

**Snapshot: COMPLETE.** Descriptive diagnostics only; no causal attribution.

Amended initial-job scope: 3360 successful conditions; the original plan had 8120 and 4760 were canceled. Canceled conditions are not completed results. The preserved root guard remains interrupted; the amendment and exact job/count checks determine this snapshot's completion status.

## Public E1 by support-document coverage

| Job | Arm | Nominal hops | Support coverage | N | EM | F1 | Selected truncation | UNKNOWN | Output cap |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beam_equal_updates | direct_text / bf16 | 3 | unannotated | 60 | 0.0% | 25.2% | 0.0% | 0.0% | 38.3% |
| hotpot_equal | direct_text / bf16 | 3 | all_support_documents | 92 | 46.7% | 56.5% | 7.6% | 20.7% | 1.1% |
| hotpot_equal | direct_text / bf16 | 3 | missing_support_documents | 36 | 19.4% | 25.9% | 19.4% | 52.8% | 0.0% |
| musique_equal | direct_text / bf16 | 3 | all_support_documents | 34 | 23.5% | 27.8% | 11.8% | 50.0% | 0.0% |
| musique_equal | direct_text / bf16 | 3 | missing_support_documents | 94 | 4.3% | 6.1% | 20.2% | 85.1% | 0.0% |
| beam_equal_updates | latent / bf16 | 3 | unannotated | 60 | 0.0% | 22.6% | 0.0% | 0.0% | 45.0% |
| hotpot_equal | latent / bf16 | 3 | all_support_documents | 92 | 37.0% | 46.9% | 7.6% | 27.2% | 0.0% |
| hotpot_equal | latent / bf16 | 3 | missing_support_documents | 36 | 16.7% | 19.5% | 19.4% | 50.0% | 0.0% |
| musique_equal | latent / bf16 | 3 | all_support_documents | 34 | 17.6% | 25.8% | 11.8% | 58.8% | 2.9% |
| musique_equal | latent / bf16 | 3 | missing_support_documents | 94 | 2.1% | 5.0% | 20.2% | 88.3% | 0.0% |
| beam_equal_updates | latent / int8 | 3 | unannotated | 60 | 0.0% | 23.1% | 0.0% | 0.0% | 41.7% |
| hotpot_equal | latent / int8 | 3 | all_support_documents | 92 | 37.0% | 46.9% | 7.6% | 27.2% | 0.0% |
| hotpot_equal | latent / int8 | 3 | missing_support_documents | 36 | 16.7% | 21.9% | 19.4% | 47.2% | 0.0% |
| musique_equal | latent / int8 | 3 | all_support_documents | 34 | 17.6% | 25.9% | 11.8% | 58.8% | 0.0% |
| musique_equal | latent / int8 | 3 | missing_support_documents | 94 | 2.1% | 4.3% | 20.2% | 89.4% | 0.0% |
| beam_equal_updates | text / bf16 | 3 | unannotated | 60 | 0.0% | 27.2% | 0.0% | 0.0% | 26.7% |
| hotpot_equal | text / bf16 | 3 | all_support_documents | 92 | 42.4% | 52.7% | 7.6% | 30.4% | 0.0% |
| hotpot_equal | text / bf16 | 3 | missing_support_documents | 36 | 25.0% | 32.7% | 19.4% | 41.7% | 0.0% |
| musique_equal | text / bf16 | 3 | all_support_documents | 34 | 26.5% | 33.7% | 11.8% | 44.1% | 0.0% |
| musique_equal | text / bf16 | 3 | missing_support_documents | 94 | 9.6% | 13.0% | 20.2% | 68.1% | 0.0% |

Supporting IDs do not establish that the answer text survived truncation. JSON includes separate truncation strata and intersected document IDs.

## Matched BF16 latent versus direct text

- beam_equal_updates, 3 nominal hops: 60 matched pairs; 0 unpaired; 0 evidence mismatches. EM disagreement IDs: none. Prediction/F1 differences are retained in JSON.
- hotpot_equal, 3 nominal hops: 128 matched pairs; 0 unpaired; 0 evidence mismatches. EM disagreement IDs: 5a72814f5542994cef4bc2eb, 5a7605f85542994ccc91868d, 5a77c15f5542997042120b1c, 5a7fb17c5542994857a767bb, 5a7fc41455429969796c1b51, 5a877fa45542993e715abf84, 5a8bd49d5542997f31a41dd7, 5a8e1027554299653c1aa15f, 5ab3010455429976abd1bc15, 5ab53436554299637185c504, 5ab6b59f5542995eadef0065, 5aba14195542994dbf0198a9, 5abb61c0554299642a094a86, 5abff0645542994516f45539, 5ac1a6bf5542991316484b8d, 5add44f35542997545bbbd0c, 5ae08dad55429945ae9593b9, 5ae10df355429920d52342ac, 5ae135fb55429920d523431f, 5ae1f70c5542997f29b3c1c5, 5ae27f22554299495565da8d, 5ae3f1bb5542995ad6573cbe, 5ae45b6b55429970de88d943, 5ae666d35542991bbc9760cc. Prediction/F1 differences are retained in JSON.
- musique_equal, 3 nominal hops: 128 matched pairs; 0 unpaired; 0 evidence mismatches. EM disagreement IDs: 2hop__31091_31122, 2hop__444265_82341, 2hop__555792_30351, 3hop1__131820_59747_60748, 4hop1__342858_131850_159767_81096, 4hop1__726675_508773_85832_745702. Prediction/F1 differences are retained in JSON.

## Synthetic E2: chain length and actual execution

| Job | Arm | Chain length | Nominal hops | Executed evidence hops | N | EM | F1 | UNKNOWN | Output cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## E4: same questions at all four ratios

| Job | Nominal hops | Ratio | Matched N | EM | F1 | EM difference vs 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

Available-arm denominators, incomplete questions, and matched per-question scores are retained in JSON.

## Limits

- Posthoc descriptive diagnostics only; these strata do not establish causal explanations.
- Supporting-document ID coverage does not prove that answer-bearing text survived truncation or was used.
- Selected truncation means evidence_ids intersect truncated_document_ids; unselected truncated documents do not count.
- UNKNOWN is the exact case-insensitive stripped sentinel; prose abstentions are not counted. Output-cap rate concerns final answers only.
- Only successful repeat-zero records enter accuracy denominators; missing conditions and errors are disclosed, not imputed.
- BEAM EM/F1, if present, describe lexical overlap, not its behavioral rubric score.
- Synthetic E2 keeps chain length, nominal budget, and actual executed evidence hops separate; evidence exhaustion is not adaptive-search failure.
- Executed evidence hops is the source record's nonempty evidence-batch count; direct_text still processes its selected evidence in one final prompt, not multiple agent steps.
- E4 paired comparisons require all four ratios and identical ordered evidence IDs for the same question and condition.
