# Topic boundary annotation guidelines

These guidelines define the human reference annotations used to evaluate Topic segmentation in
the 20-program WBS and NHK News 7 test set.

## Annotation unit

A Topic is one news subject: one event, policy, organization-level development, accident, match,
weather segment, market segment, or product/service feature that a viewer would recognize as a
single news item. Titles and domains help identify the item but are not used by the boundary
metrics.

Annotators determine boundaries from the broadcast video. System predictions are not shown during
annotation.

## When to start a new Topic

Start a new Topic when at least one of the following occurs:

- the subject changes to a different event, organization, person, or issue;
- the program explicitly introduces the next news item or changes to a distinct segment;
- a short-news or headline block changes to a new item;
- after a commercial, promotion, or sponsor interval, the editorial content resumes with a
  different subject.

Do not start a new Topic for a camera cut, a presenter change, or a change between studio reading,
field footage, interviews, and graphics when the subject remains the same. Consecutive results or
highlights from the same competition or league form one Topic. A related report with a different
editorial focus is a separate Topic.

If a commercial or other non-editorial interval interrupts a report and the same subject resumes,
annotate one Topic spanning the interruption. If the subject changes, leave the non-editorial
interval as a gap between Topics.

## Boundary times

- Record seconds from the beginning of the recording; decimals are allowed.
- `start_sec` is the earliest onset of the Topic in speech, captions, or visuals.
- `end_sec` is the latest end of the Topic in speech, captions, or visuals.
- Use caption text as the canonical wording when captions are available, while using audio and
  visuals to resolve the exact time.
- Do not include adjacent commercials, promotions, sponsor credits, or station presentation unless
  they interrupt a Topic that continues with the same subject.
- When the transition is ambiguous, record the best boundary and document its uncertainty.

## Boundary types

Each Topic start has one structured boundary type in its `notes` field:

- `program_edge`
- `studio_transition`
- `vtr_transition`
- `bumper_transition`
- `cm_adjacent`
- `continuous_read`

The stored form is `[境界タイプ: <value>]`.

## Topic records

Topic identifiers are assigned in chronological order (`t001`, `t002`, ...). Each Topic contains:

- `topic_id`
- `start_sec`
- `end_sec`
- an optional IPTC top-level domain
- a short label used for annotation review
- the structured boundary-type tag

## Quality checks

- Every Topic has `start_sec < end_sec`.
- Topic identifiers are unique and ordered by `start_sec`.
- Topic spans do not overlap.
- Short news items remain separate Topics when their subjects differ.
- Format changes alone do not create Topic boundaries.
- Non-editorial gaps are not labeled as Topics.
- Ambiguous decisions are reviewed against the video before finalization.
