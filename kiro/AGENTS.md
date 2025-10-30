# AGENTS Contributor Guide

## Purpose
This guide keeps local contributors aligned on how TheCatBouncer works, how to set up a development environment, and what quality bar to meet before handing changes to the rest of the team.

## Project Snapshot
- **Goal**: Identify the household cat while deterring intruders via light and sound automation.
- **Core pipeline**: `main-file_and_active_analysis_pipeline.py` orchestrates passive monitoring, active YOLO inference, colour analysis, and deterrent triggers.
- **Supporting modules**: `color_analyzer.py`, `passive_analyzer.py`, `hue_controller.py`, `data_manager.py`, `utility_recorder.py`, and helpers inside `HelpingProgramms/`.
- **Key assets**: Configuration via `config.ini`, model files under `yolo11_*`, audio deterrents in `cat_scare_sound/`, and diagnostic output in `CatDetectorData/`.
- **Local-only note**: The optional `.kiro/` workspace folder contains editor state; no steps in this guide depend on it.

## Environment Setup
1. Install Conda (Miniconda or Anaconda).
2. Create the project environment: `conda env create -f environment.yml`.
3. Activate it with `conda activate cat-analysis-env`.
4. Install optional vendor SDKs (e.g., GPU drivers, Philips Hue bridge pairing tools) as required by your hardware.
5. Keep `environment.yml` in sync with any dependency changes; update and document pip-only packages inside the embedded `pip:` block.

## Local Development Workflow
1. Copy the project folder to a new timestamped workspace before making changes (e.g., `TheCatBouncer_2025-10-15`).
2. Track changes in a simple changelog file (e.g., `DEV_NOTES.md`) so other collaborators can replay your steps.
3. When editing code, keep modules focused: create new helper files rather than merging unrelated logic into large scripts.
4. When work is ready to share, zip the updated folder or sync it via the agreed local medium (USB drive, shared NAS, etc.) and include your changelog.

## Coding Standards
- Python 3.10 syntax, prefer type hints for new functions.
- Format with `black .` and lint with `ruff .` before packaging your changes.
- Follow existing logging patterns in `utility_recorder.py` instead of ad-hoc prints.
- Centralise configuration in `config.ini`/`config.sample.ini`; avoid hard-coded paths or credentials.
- Respect platform portability; code must work on Windows, macOS, and Linux.

## Testing & Verification
- Automated tests are not yet in place; add them when contributing new logic (pytest is preferred if introduced).
- At minimum, run a dry integration test: execute the main pipeline with a short video or webcam session and inspect logs in `CatDetectorData/SystemLogs/`.
- Validate deterrent actions by using stub or simulation modes when hardware is unavailable.
- Log manual test steps and outcomes in your changelog so others can repeat them locally.

## Working With Data & Assets
- Keep large media, generated logs, and backups out of the shared handover package; store them on local storage only.
- Place reusable sample assets for tests inside a dedicated `samples/` directory with small, anonymised files.
- Clean `CatDetectorData/` and other runtime folders before distributing your workspace copy.

## Documentation Expectations
- Update `readme.md`, `config.sample.ini`, or module docstrings when behaviour changes.
- Add quick-start notes for new scripts inside `HelpingProgramms/` if you introduce them.
- Record environment changes in `environment.yml` comments so other agents understand why dependencies were added.
- Summarise key changes in your `DEV_NOTES.md` (or equivalent) to keep offline collaboration transparent.

## Quality Checklist Before Sharing
- [ ] Workspace copy labelled with date/version.
- [ ] Code formatted with `black` and linted with `ruff`.
- [ ] Tests and manual verification steps documented with relevant artefacts (log snippets, screenshots).
- [ ] Configuration and documentation updates included when behaviour changes.
- [ ] No secrets, large binaries, or platform-specific paths placed in the handover package.

## Communication & Support
- Use the agreed local channel (e.g., team chat, phone, or shared notebook) to discuss feature ideas or flag blockers.
- Surface architectural or dependency changes early so others can plan their local setup updates.
- Capture open questions or follow-up tasks in your changelog to keep the offline handover smooth.

## Ready To Contribute?
Review the checklist, run the linters, and package your updated workspace. Share it through the local channel; maintainers monitor incoming bundles and will help unblock you. Thanks for helping keep TheCatBouncer purring along.

