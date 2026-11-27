# Data

Nothing in this tree is tracked by git. The folders and their README files are tracked so the
layout survives a clone.

| Folder | Holds |
| --- | --- |
| `raw/` | data as delivered or downloaded, untouched |
| `interim/` | reshaped intermediates, safe to regenerate |
| `derived/` | analysis outputs and final cubes |
| `inventory/` | catalogue records of what data exists, each with its retrieval date |

Every dataset note states its date: acquisition window for imagery, declared epoch for model
output, retrieval date for a catalogue record. If no date is published, say so.
