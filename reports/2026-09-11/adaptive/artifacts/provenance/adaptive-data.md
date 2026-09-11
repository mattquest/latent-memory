# Global MuSiQue retrieval data

The adaptive comparison uses 21,100 unique paragraphs from all 2,417 questions in the official MuSiQue v1.0 answerable development split. The retrieval corpus contains only `id`, `title`, and `text`. IDs are SHA-256 digests of exact title/text pairs; corpus order follows those digests. Source question IDs and paragraph indices live in a separate provenance ledger, never in the retrieval index.

The local development split has six questions, two each with two, three, and four hops. The local test split has 96 questions, 32 per hop count. Both exclude all 128 question IDs and exact question texts from the original experiment matrix. Questions were ranked by SHA-256 of `20260913:source_id` within structural hop groups; two development questions precede the 32 test questions in each group. Selection reads no model outputs or experiment results. Development and test IDs and exact question texts are disjoint.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `data/adaptive_musique_corpus.jsonl` | 21,100 | `4e8ad63e12ab37e7fabec0158241d890a96171e006d226a7f22b6d00d594ce4a` |
| `data/adaptive_musique_corpus_provenance.jsonl` | 21,100 | `ec8ce692b2f8294d641441c993c72d22a05e3d0a9eade1f6a9562b30b5869a3b` |
| `data/adaptive_musique_dev.jsonl` | 6 | `a7627e89d17ee6933b6b810406784dbc9e115264432dcf4df34b3fbce09f87a3` |
| `data/adaptive_musique_test.jsonl` | 96 | `e09e7d8e3aa87cd81bb05d055013c9e9ab6e97ed7688a249ba6dd4563b53124f` |

The 12.94 MB corpus contains 2,518,129 Qwen3 tokenizer tokens when encoding each title, newline, and paragraph without special tokens. The tokenizer JSON hash and exact counting convention are in `data/adaptive_musique.manifest.json`; these are content tokens, not final chat/context overhead. Preparation used the tokenizer on CPU and did not load model weights.

Deduplication removes 27,215 exact repeated paragraph occurrences. It retains distinct text variants: 1,090 titles have more than one distinct paragraph. Of 233 unique support paragraphs for the selected questions, 173 occur in more than one original question context. Disjoint question IDs do not imply disjoint facts, articles, or freedom from public pretraining exposure. This is global retrieval over the supplied development paragraphs, not full-Wikipedia retrieval or the publisher's hidden test set.

Question records retain original answers and aliases, `supporting_context_ids` mapped to corpus IDs, and structural hop counts. Those fields are scoring/provenance metadata. A retriever or controller may read only the question and retrieved corpus content. No decomposed questions or intermediate answers are included in these new question records.

Exact reproduction from the existing local download:

```sh
.venv/bin/python scripts/prepare_adaptive_musique.py \
  --archive data/raw/musique.zip \
  --exclude data/musique_dev_sample.jsonl \
  --output-dir data \
  --tokenizer-json models/Qwen3-8B/tokenizer.json
```

The script requires the original archive SHA-256 `98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`. Its default seed and sample sizes match the released artifacts. A separate output-directory replay produced byte-identical corpus, provenance, development, and test JSONLs. All 6,404 support occurrences in the source development split map to the deduplicated corpus, and supporting flags agree with decomposition paragraph annotations.

MuSiQue data are attributed to Trivedi et al. (2022) and distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) through the [authors' repository](https://github.com/StonyBrookNLP/musique). The [official v1.0 archive](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view) is linked by the download script at revision `922ac98f19a201998dbdae6d7f2887a5258dbdeb`. Normalization, global deduplication, new IDs, and local split selection are modifications made here. The data license applies separately from project code.
