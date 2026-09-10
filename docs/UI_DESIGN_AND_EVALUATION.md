# Research Harness UI and Evaluation Plan

## Purpose

The UI should make one comparison clear: a predictable baseline versus a more
autonomous agentic system. It should look professional without exposing the
internal tool graph or hidden chain-of-thought.

## User story

As a researcher, I enter a question about the uploaded paper corpus. The UI
shows two possible research paths. I select one path, choose Fast or Slow
execution where applicable, and start one run. The selected path returns a
short evidence-backed answer, supporting passages, and extracted triples. In
parallel, the system reports whether the question was easy or conceptually
ambiguous, whether the triples are supported, and lets me give brief feedback.

## Minimal UI

1. **Question bar** — one large search field and a `Research` action.
2. **Two vertical path cards** — exactly one can be selected:
   - **Fast baseline**: predictable retrieval, top passages, and direct LLM
     extraction. Suitable for direct questions such as “Who authored this
     paper?”
   - **Agentic research**: the system interprets the goal, identifies the
     concepts and comparison criteria, chooses retrieval/specialist actions,
     checks intermediate results, and then produces the answer. It is slower
     because it performs adaptive planning and verification.
3. **Mode selector** — `Fast` and `Slow` are execution budgets, not additional
   research paths. Fast uses fewer calls; Slow allows deeper planning and
   verification.
4. **Result view** — answer first, triples second, evidence expandable.
5. **Evaluation panel** — `Concept interpretation`, `Triple quality`, and
   `User feedback`. Show simple statuses, not internal traces by default.

Only the selected path executes. A failed run remains visible as an incomplete
result and offers `Try deeper research`; it does not silently run every path.

## Terminology

- **Research path**: the selected strategy (Fast baseline or Agentic research).
- **Execution mode**: the call/latency budget (Fast or Slow).
- **Concept interpretation**: the entities, properties, comparison criteria,
  aggregation, and scope the system believes are required by the question.
- **Concept ambiguity**: multiple plausible interpretations of those concepts;
  it is not the same as model error.
- **Triple quality**: whether subject–predicate–object claims are correct,
  relevant, and supported by supplied evidence.
- **Disagreement**: disagreement among independently generated concept plans
  or interpretations. It is a diagnostic signal, not an accuracy score by
  itself.

## Evaluation recommendation

There is no defensible universal scalar for “understanding” an open-ended
question. Use a small annotated benchmark and report separate metrics:

### Primary: concept/decomposition score

For each question, annotate the required concepts and steps. Compare the
system output with the annotation using:

- **Concept precision, recall, and F1** over normalized concept/criterion
  items (for example `author`, `success metric`, `corpus scope`, `ranking`).
- **Step/operator F1** over a QDMR-like decomposition: selection, filtering,
  comparison, aggregation, and ranking steps.
- **Reference/dependency accuracy**: whether a later step uses the correct
  earlier result.

QDMR is a useful model because it represents a complex question as ordered
steps with references and operators, and the Break dataset provides annotated
decompositions. See the [QDMR paper](https://aclanthology.org/2020.tacl-1.13/)
and [official Break repository](https://github.com/allenai/Break).

### Secondary: disagreement as a diagnostic

Run a small number of independent concept plans (different samples or models),
normalize their concept sets, and report:

```text
disagreement = 1 - average pairwise Jaccard similarity
```

Use this only as an ambiguity/instability indicator. Calibrate it against
human adjudication on a small set; low disagreement does not prove that the
interpretation is correct. AmbigQA is a useful reference for collecting
multiple acceptable interpretations/answers rather than forcing one answer;
see its [paper implementation](https://github.com/shmsw25/AmbigQA).

### Triple and answer quality

- Triple precision, recall, and F1 against human-annotated gold triples.
- Evidence-support rate: percentage of output triples with at least one
  correct supporting passage.
- Answer usefulness: a single human rating (`Useful`, `Partially useful`, or
  `Not useful`).

## Example

Question: “Across this corpus, which author is most successful?”

Expected concepts might include `author`, `corpus`, `success criterion`,
`aggregation`, and `ranking`. The system should not silently invent the
criterion. It should either use a supplied criterion (for example citation
count) or state that the question is under-specified and ask/declare an
assumption. The evaluation records whether the concept plan captured this
ambiguity and whether the final triples support the ranking.

## Relevant implementation references

- [Break / QDMR](https://github.com/allenai/Break): annotated question
  decompositions and logical forms.
- [AmbigQA](https://github.com/shmsw25/AmbigQA): multiple acceptable answers
  and ambiguity-aware evaluation scripts.
- [OneKE](https://github.com/zjunlp/OneKE): schema-guided extraction,
  triple extraction, and KG visualization patterns.

These projects are references for evaluation/data formats, not dependencies to
add automatically. Keep the first experiment small: 20–50 questions, with two
or three human annotations for the ambiguous subset.

## Open decisions

- Exact annotation schema for concepts and QDMR-like steps.
- Number of independent concept plans used for disagreement.
- Whether “success” questions require the user to select a metric or allow the
  system to propose one and request confirmation.
- Annotation budget and adjudication procedure.
- Whether Slow mode is the default for ambiguous questions.
