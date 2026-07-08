# Event generators (GENIE & NuWro)

This project uses Docker to run GENIE and, later, NuWro for neutrino-event generation alongside the analytic cross-section work in `plot_cross_section.py`.

## GENIE Docker Setup

Docker is being used instead of a native install because the GENIE stack is easier to reproduce this way: the compiler toolchain, ROOT, LHAPDF, and related libraries are captured in the image rather than hand-installed on macOS. That makes the setup easier to rebuild, easier to verify, and easier to continue later on the same machine or another machine.

Current confirmed GENIE setup:

- Docker image name: `nuclearproject-genie`
- GENIE release/tag: `R-3_06_02 / v3.06.02`
- GENIE source path inside Docker: `/opt/genie-generator`
- GENIE install path inside Docker: `/opt/genie-install`
- ROOT is installed through micromamba
- ROOT version confirmed: `6.40.02`
- LHAPDF is installed because GENIE's HEDIS build requires `LHAPDF/LHAPDF.h`
- GENIE config/tune path: `/opt/genie-generator/config`
- `GXMLPATH` must point to `/opt/genie-generator/config`
- Default GENIE tune loads successfully: `G18_02a_00_000`

Confirmed commands available:

- `/opt/genie-install/bin/gevgen`
- `/opt/genie-install/bin/gevgen_atmo`
- `/opt/genie-install/bin/gxscomp`
- `/opt/genie-install/bin/gevdump`

Enabled GENIE features in this Docker setup:

- flux drivers
- geometry drivers
- atmospheric event generation app

Disabled features in this Docker setup:

- `PYTHIA6`
- `PYTHIA8`
- `LHAPDF5`
- `Geant4 interface`
- `INCL++`

Important runtime note:

- `GXMLPATH` must be set to `/opt/genie-generator/config` so GENIE can find `Messenger.xml` and the `G18_02a_00_000` tune directory.
- The `FATAL` messages from `gevgen` and `gevgen_atmo` during help-style tests are expected if no runtime inputs are provided.
- The important success signal is that GENIE loads `Messenger.xml`, loads `G18_02a_00_000`, and prints syntax/help text before complaining about missing runtime inputs.
- `gevgen` printing errors like `Unspecified neutrino energy` during a test is okay.
- `gevgen_atmo` printing errors about missing flux, geometry, cross-section XML, or exposure during a test is okay.

## Docker and GENIE Command Cheat Sheet

### A. Normal Mac terminal commands

Check current folder:

```bash
pwd
```

List files:

```bash
ls
```

Check Docker version:

```bash
docker --version
```

Check Docker works:

```bash
docker run hello-world
```

Build or rebuild the GENIE Docker image:

```bash
docker build --no-cache -f docker/Dockerfile.genie -t nuclearproject-genie .
```

Enter the GENIE Docker container from the `NuclearProject` folder:

```bash
docker run -it --rm -v "$PWD":/work -w /work nuclearproject-genie bash
```

### B. How to know if I am inside Docker

Mac terminal prompt typically looks like:

```text
(.venv) haatim@Haatims-MacBook-Air NuclearProject %
```

Docker prompt typically looks like:

```text
(base) root@container_id:/work#
```

or:

```text
root@container_id:/work#
```

### C. Commands inside Docker

Check mounted project files:

```bash
ls /work
```

Move to project folder:

```bash
cd /work
```

Move to GENIE source:

```bash
cd /opt/genie-generator
```

Move to GENIE install folder:

```bash
cd /opt/genie-install
```

Check GENIE environment variables:

```bash
echo $GENIE
echo $GXMLPATH
```

Set GENIE environment manually if needed:

```bash
export GENIE=/opt/genie-install
export GENIE_SOURCE=/opt/genie-generator
export PATH=$GENIE/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=$GENIE/lib:/opt/conda/lib:$LD_LIBRARY_PATH
export GXMLPATH=/opt/genie-generator/config
```

Check ROOT:

```bash
which root-config
root-config --version
```

Check LHAPDF:

```bash
which lhapdf-config
lhapdf-config --version
```

Check GENIE commands:

```bash
which gevgen
which gevgen_atmo
which gxscomp
which gevdump
```

Test GENIE:

```bash
gevgen --help | head -40
gevgen_atmo --help | head -40
```

Run the install check script:

```bash
bash scripts/check_genie_install.sh
```

View the saved install proof:

```bash
cat outputs/genie/genie_install_check.txt
```

### D. How to exit Docker

Use:

```bash
exit
```

After exiting, the prompt should return to something like:

```text
(.venv) haatim@Haatims-MacBook-Air NuclearProject %
```

## Verification steps

Use these exact commands to rebuild and verify the setup:

```bash
docker build --no-cache -f docker/Dockerfile.genie -t nuclearproject-genie .
docker run -it --rm -v "$PWD":/work -w /work nuclearproject-genie bash
bash scripts/check_genie_install.sh
cat outputs/genie/genie_install_check.txt
```

The install check script is safe to run from inside Docker at `/work`. It creates `outputs/genie/` if needed and writes a reproducible install log to `outputs/genie/genie_install_check.txt`.

## Troubleshooting

- If GENIE says `Cannot find root-config`, ROOT is missing or `PATH` is wrong.
- If GENIE says `LHAPDF/LHAPDF.h: No such file or directory`, LHAPDF is missing from `docker/Dockerfile.genie`.
- If GENIE says `Messenger.xml: No such file or directory`, `GXMLPATH` is not set.
- If GENIE says `No valid tune directory associated with G18_02a_00_000`, `GXMLPATH` is wrong or the config folder is missing.
- If `gevgen` says `Unspecified neutrino energy`, that is okay during a test because it means GENIE loaded but no runtime inputs were given.
- If `gevgen_atmo` says flux, geometry, or exposure inputs are missing, that is okay during a test because it means the atmospheric app loaded but no runtime inputs were given.

## Persistence warning

- Files saved inside `/work` are saved to the Mac project folder because `/work` is the mounted repository.
- Files saved outside `/work`, such as `/opt/genie-generator` or `/opt/genie-install`, exist only inside the Docker container or image.
- Because `docker run` uses `--rm`, manual changes inside a running container can disappear after exit unless they are captured in the Dockerfile and the image is rebuilt.
- That is why LHAPDF must be written into `docker/Dockerfile.genie`, not only installed manually inside a temporary running container.

## Output layout

Keep generator artifacts under fixed paths:

- `outputs/genie/` for GENIE logs, checks, cards, and generator output
- `outputs/nuwro/` for future NuWro output
- `data/flux/` for atmospheric or other flux input files

## NuWro setup

NuWro is still planned as a follow-up generator in this project. When that setup is added, keep the same Docker-first workflow and save outputs under `outputs/nuwro/`.

## Version and tune record

Record exact versions and image details here whenever the setup changes:

| Component | Version / tune | Docker image | Date noted | Notes |
|-----------|----------------|--------------|------------|-------|
| GENIE | `R-3_06_02 / v3.06.02` | `nuclearproject-genie` | _update when rebuilt_ | Default tune confirmed to load: `G18_02a_00_000` |
| ROOT | `6.40.02` | `nuclearproject-genie` | _update when rebuilt_ | Installed through micromamba |
| NuWro | _TBD_ | _TBD_ | _TBD_ | Not installed yet |
