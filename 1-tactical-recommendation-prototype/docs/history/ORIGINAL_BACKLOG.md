# Future Work: Proper Geopolitical Fine-Tuning

## Historical status (superseded by the current release checklist)

The current QLoRA adapter is a proof of concept trained on 20 mock training examples. It verifies that the pipeline can load Qwen, train an adapter, produce structured JSON, and evaluate extraction quality. It is not yet a production-quality military or geopolitical adapter.

## Required two-stage training plan

### 1. Domain adaptation

Collect a large, legally usable corpus of English, Arabic, and mixed-language geopolitical reporting, especially material related to the MENA region. The corpus should expose the base model to military systems and weapons, countries and ministries, leaders and officials, defence companies and organisations, Arabic names, transliterations, abbreviations, aliases, geopolitical relationships, and topics.

Use the corpus for continued pretraining or domain-adaptive training. This teaches the model the language and terminology of the domain, but does not by itself teach the required JSON schema.

### 1.1 Tokenizer and domain-term evaluation

Evaluate the base Qwen tokenizer on recurring domain terms before changing it. Record how
many tokens are used for important Arabic, English, and mixed-language entities, for example:

```text
Lockheed Martin | لوكهيد مارتن | LMT
Mohammed bin Salman | محمد بن سلمان | MBS
MQ-9B SkyGuardian | إف-35 | THAAD
```

It is normal for `لوكهيد مارتن` to be represented by several tokens. Fine-tuning does
not automatically make it one token: Qwen's existing tokenizer vocabulary stays fixed.
Domain-adaptive training can still teach the model that the *sequence* of tokens refers
to Lockheed Martin and how it relates to Saudi Arabia, defence systems, and contracts.

If evaluation shows that frequent critical terms are handled poorly, test a controlled
tokenizer-vocabulary extension or retrained domain tokenizer. That work requires resizing
the model's input/output embeddings and then continued training, so it must be compared
against the original tokenizer on held-out Arabic, English, mixed-language, and unseen-term
tests. Do not add every rare name as a custom token; reserve this for frequent, high-value
terminology where evaluation proves a benefit.

### 2. Structured extraction fine-tuning

Create reviewed article-to-JSON examples:

```text
article text -> countries, companies, organisations, profiles, systems, topics, relationships
```

Train a QLoRA adapter on these labeled examples. The examples must include English, Arabic, mixed-language text, aliases, transliterations, abbreviations, and unseen entities. Keep a fixed validation set and a completely untouched test set.

After the adapter passes the held-out test set, update the Tactical Report model prototype to run Qwen + LoRA for every selected incoming article. Replace saved mock extraction outputs and prewritten article tags with versioned live inference records that display the generated JSON, model version, confidence or review status, and any unresolved entities. A cached result is acceptable after it has been generated; a manually prepared result is not a substitute for inference.

### 3. Learned entity linking and enrichment

The current alias catalogue is only a baseline and must not be treated as the final intelligence layer. Build a learned entity-linking component that can identify that different spellings, transliterations, abbreviations, and contextual mentions refer to the same canonical person, company, organisation, country, or system.

Train and evaluate it on reviewed entity-linking examples. Candidate generation may use multilingual embeddings, while the final choice should use article context and confidence scoring. Keep the catalogue as the auditable canonical-ID store and fallback, but let the learned component discover likely matches and propose unknown entities for review instead of requiring every alias to be manually entered first.

After the learned linker passes its evaluation, replace the manual catalogue-matching path in the Tactical Report model prototype. Update the live recommendation flow, the A-to-Z walkthrough, and the prototype screens so every selected incoming article is linked by the trained component and the UI clearly shows its confidence and review status. The manual catalogue remains only as an auditable fallback for low-confidence cases.

## Target pipeline

```text
Open geopolitical corpus
    -> domain-adapt the pretrained model
    -> create reviewed extraction examples
    -> train the QLoRA extraction adapter
    -> learned entity linking and enrichment
    -> test on unseen articles
    -> send extracted filters to the recommender
```

## Important distinction

Raw internet articles teach domain familiarity. They do not reliably teach the model which text belongs in each structured field. Labeled extraction examples are still required.

## Data and safety requirements

- Prefer open or permissively usable sources and retain source metadata.
- Do not use private, paid, or restricted Tactical Report content without authorization.
- Review and clean articles before adding them to training data.
- Do not train automatically on every model prediction; use a reviewed queue and periodic dataset versions.
- Compare every new adapter against the same fixed test set before deployment.

## Not part of this future task

The multilingual embedding model used by the recommendation layer does not need to be fine-tuned first. Start with exact entity/filter matching plus pretrained multilingual embeddings, then fine-tune the recommender only after collecting real relevance or click feedback.
