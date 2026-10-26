# Recovery and restart

The 2026-10-26 scaling run stopped after recorded update 489/512. The last saved adapter was checkpoint-448 (adapter_config.json, adapter_model.safetensors, README.md only). No optimizer, scheduler or RNG state exists, so exact continuation is impossible. The cause of process disappearance is unknown. Preserve the old run and restart from the original pinned base model, using all 4096 cached samples and the same seed, learning rate, objective and one-epoch schedule. This retains the planned comparison; no adapter warm start is mixed into it.

New checkpoints publish atomically after adapter and training state are written, every 16 optimizer updates for the restart profile. State includes AdamW moments, learning-rate scheduler, CPU/CUDA/Python RNG, next shuffled-row index, completed updates and loss history, with config/manifest verification and state checksum. A .partial directory is never a valid resume source. --resume accepts complete checkpoints only and requires a new output directory. Example: python scripts/train_sft.py --config configs/sft-v3-restart.json --resume <complete-checkpoint> --output <new-output>. Resume evaluation must use the resulting adapter directory, not the interrupted adapter path.

Tests verify exact CPU toy-model next-update equivalence, reject adapter-only checkpoints, reject changed protocol and corrupt state. They do not guarantee bitwise GPU determinism across CUDA/software changes. No PyTorch full model pickle or downloaded pickle checkpoint is loaded; training state is self-created and loaded with weights_only=True.

Restart pipeline: python scripts/run_scaling_restart.py. It audits cached data, trains, then evaluates the original 128 threshold-selection rows and 372 reused development rows. Official test remains untouched. Keep Windows awake while training.


## Registered step-112 recovery (v0.6.2)

The complete checkpoint is `D:/career-projects/runs/MemeGuard/20261026-scale-restart-v1/training/checkpoint-112`. The new output is `D:/career-projects/runs/MemeGuard/20261026-scale-resume112-v1`. The runner requires step 112 / example 896, validates configuration and manifest identity, and records checksums before launching training. It refuses to overwrite an existing output directory.

Use `scripts/start_scale_resume.ps1 -Checkpoint <complete-checkpoint> -Output <new-directory>` from the Windows VS Code terminal. It calls the saved Python runner with the dedicated CUDA interpreter. Training is followed by the existing 128-row threshold-selection and 372-row reused-development evaluation. No official test evaluation is triggered.

The runner requests Windows SYSTEM_REQUIRED only during execution and clears it on exit. Screen-off and lock are allowed; keep the Windows account signed in and avoid manual sleep, shutdown, or reboot. This is not a persistent power-policy change. Keep this VS Code remote terminal connected for this run: Task Scheduler returned exit code 1 without producing a Python log on this host, so a session-independent launch has not been established. The failed task has no recurring trigger and is disabled.

The receipt confirms checkpoint validation; actual optimizer restoration and progress must additionally be verified in training logs/status. GPU bitwise identity with uninterrupted training is not guaranteed. A resumed run is not counted as another independent training seed.
