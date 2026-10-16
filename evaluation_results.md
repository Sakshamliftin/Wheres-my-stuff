# Evaluation results

Computed from the current `observations.db` log. These are not MOTA, IDF1, or HOTA — there is no tracking ground truth.

Observations: **3679**

## Detections per class

| Class | Detections |
| :--- | ---: |
| laptop | 1821 |
| glasses | 983 |
| chair | 431 |
| television | 337 |
| computer mouse | 107 |

## Zone dwell

Share of that class's detections that fell inside each zone.

| Class | Zone | Detections | Share |
| :--- | :--- | ---: | ---: |
| chair | Towel Top-Right | 416 | 97% |
| chair | Towel Top-Left | 15 | 3% |
| computer mouse | Towel Top-Right | 59 | 55% |
| computer mouse | Towel Bottom-Right | 40 | 37% |
| computer mouse | Towel Bottom-Left | 7 | 7% |
| computer mouse | Towel Top-Left | 1 | 1% |
| glasses | Towel Top-Right | 919 | 93% |
| glasses | Towel Bottom-Right | 48 | 5% |
| glasses | Towel Top-Left | 16 | 2% |
| laptop | Towel Top-Left | 1503 | 83% |
| laptop | Towel Bottom-Right | 225 | 12% |
| laptop | Towel Top-Right | 89 | 5% |
| laptop | Towel Bottom-Left | 4 | 0% |
| television | Towel Top-Left | 333 | 99% |
| television | Towel Bottom-Left | 4 | 1% |

## Re-identification

- Global tracks: 21
- Distinct ByteTrack IDs: 46
- Tracks that absorbed a later ByteTrack ID: 13
- Stitched ByteTrack IDs (extra IDs beyond the first): 25
- Tracker backend recorded for this run: `clip`
- Stitches counted while tracking: 47

## Sample questions

Same prompts as the Task 5 report. `resolved` means the class parser hit a logged class. `visual match` means the FAISS index answered. `not observed` is the expected graceful miss when the item was never seen.

| Question | Status | Class | Zone | Retrieval |
| :--- | :--- | :--- | :--- | :--- |
| Where are my keys? | not observed | keys | — | class |
| Where did I leave my wallet? | not observed | wallet | — | class |
| Where are my spectacles? | resolved | glasses | Towel Top-Right | class |
| Where are my earphones? | not observed | earphones | — | class |
| Where is my watch? | not observed | watch | — | class |
| Where is my cream bottle? | not observed | cream bottle | — | class |
| Where is my laptop? | resolved | laptop | Towel Top-Left | class |
| Where is the TV? | resolved | television | Towel Top-Left | class |
| Where is my chair? | resolved | chair | Towel Top-Right | class |
| Where is my water bottle? | not observed | water bottle | — | class |
| Where is my mouse? | resolved | computer mouse | Towel Top-Right | class |
| Where is my laptop charger? | not observed | laptop charger | — | class |
| Where is my mobile charger? | not observed | mobile charger | — | class |
| What objects do you see? | summary | — | — | class |
| Where is my passport? | not observed | passport | — | class |
