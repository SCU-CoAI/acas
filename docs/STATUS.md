# Artifact status

We apply for the **Available**, **Reviewed**, and **Reproducible** badges.

## Available

The artifact is complete and self-contained: the four ACAS method implementations, the
dense baseline, every operating-point configuration behind the paper's tables, and both
measurement pipelines, documented in [README.md](../README.md), [INSTALL.md](INSTALL.md),
[REQUIREMENTS.md](REQUIREMENTS.md), [INTERFACE.md](INTERFACE.md), and
[REPRODUCE.md](REPRODUCE.md), under an open-source LICENSE. The only external
downloads (model weights, pinned pip packages) are fetched by documented one-command steps.

## Reviewed

The artifact runs to produce the outputs described: each of the five C++ trees builds
with one standard CMake command, and every table row maps to one documented command
([REPRODUCE.md](REPRODUCE.md)) whose defaults are the paper's operating points.
Both pipelines were built and run end-to-end from clean clones on the target hardware.

## Reproducible

An independent party can regenerate the computational results from this artifact alone:
the accuracy tables on any CUDA Linux GPU (environments built entirely by the pinned
setup scripts, one command per row), and the efficiency tables on a Jetson AGX Orin with
the single-session sweep scripts. Hardware constraints are declared in
[REQUIREMENTS.md](REQUIREMENTS.md) and in the submission form.
