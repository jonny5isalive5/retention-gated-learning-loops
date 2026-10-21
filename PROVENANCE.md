# Publication provenance

Prepared 21 October 2026 from the owner's existing RGLL folder. The working source was read only. A separate private evidence snapshot and clean publication copy were created under the owner's designated Codex storage root. No training, reruns, algorithm edits or unrelated project changes were performed.

## Included and excluded

77 supplied research files (about 5.36 MB) were retained: source, requirements, original README, eight result directories, and eight console logs. Result directories contain modified main/stress/ablation and strict 100-fact, 1,000-fact, raised-budget and widened-network evidence. Every available failure log and historical configuration/hash is retained. `README.md` was relocated verbatim to `README.original.md`; the new README is the public overview.

The only supplied file with changed content is `strict_1000_run.log`: two local project paths and one local Python installation path were replaced by `<PROJECT>/` and `<PYTHON>/`. The traceback and encoding failure remain. Original bytes remain in the owner's untouched source and private snapshot. The original digest and public digest are both in `provenance/source-inventory.json`.

The local virtual environment and Python caches are excluded: they are generated dependencies, include installation paths, and are unnecessary for research evidence. Installed package versions were extracted from metadata without importing the experiment, and recorded in `requirements-observed.txt`. No credential patterns were found in the publication scan; a review of personal paths, email patterns, license notices and file sizes found only the redacted log paths and a legitimate dependency URL. No supplied research file exceeds 1 MB. No Git history was present or imported.

The original MS Paint drawing was supplied separately from the owner's Desktop and confirmed by the owner after the initial publication. It is included unchanged as `assets/original-concept.png`, credited to Jonathan Harriss. `provenance/drawing.json` records its original filename, size and matching original/public SHA-256. The PNG has no text or EXIF metadata chunks. Its original Desktop file remains untouched; a private preservation copy is also retained.

## Hash coverage and historical gaps

`provenance/source-inventory.json` maps every supplied research file to its public path, size, original SHA-256 and public SHA-256. `SHA256SUMS` covers every publication file except itself. `.gitattributes` disables line-ending conversion to preserve evidence bytes. `verify_evidence.py` checks those hashes and stored result consistency without training.

Historical strict configurations were copied unchanged. For every strict run, the current `rgll/strict.py`, `rgll/data.py` and `rgll/learner.py` match their historical hashes. The current `run_strict.py` matches the width-256 and width-512 configurations; it differs from the 100-fact, initial 1,000-fact and 1,000-loop configurations. The original README describes later budget and width flags. Earlier runner revisions were not found and have not been reconstructed. Modified runs have configuration text in their reports but no equivalent pre-run hash seal.

The original README records interrupted stress attempts and diagnostics for which separate raw files were not supplied. Those narratives remain available as historical claims; this repository does not assert that missing raw evidence has been recovered. Existing result plots are retained as supplied, not regenerated or independently validated pixel by pixel.
