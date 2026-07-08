#!/usr/bin/env bash

set -u

OUT_DIR="outputs/genie"
OUT_FILE="${OUT_DIR}/genie_install_check.txt"
GENIE_SOURCE_DEFAULT="/opt/genie-generator"
GENIE_INSTALL_DEFAULT="/opt/genie-install"

mkdir -p "${OUT_DIR}"

{
  echo "GENIE install check"
  echo "Timestamp: $(date)"
  echo "Working directory: $(pwd)"
  echo

  echo "GENIE source path: ${GENIE_SOURCE:-${GENIE_SOURCE_DEFAULT}}"
  echo "GENIE install path: ${GENIE_INSTALL:-${GENIE_INSTALL_DEFAULT}}"
  echo

  if [ -f "${GENIE_SOURCE_DEFAULT}/VERSION" ]; then
    echo "GENIE version file (${GENIE_SOURCE_DEFAULT}/VERSION):"
    cat "${GENIE_SOURCE_DEFAULT}/VERSION"
  else
    echo "GENIE version file not found at ${GENIE_SOURCE_DEFAULT}/VERSION"
  fi
  echo

  echo "root-config version:"
  if command -v root-config >/dev/null 2>&1; then
    root-config --version
  else
    echo "root-config not found"
  fi
  echo

  echo "lhapdf-config version:"
  if command -v lhapdf-config >/dev/null 2>&1; then
    lhapdf-config --version
  else
    echo "lhapdf-config not found"
  fi
  echo

  for cmd in gevgen gevgen_atmo gxscomp gevdump; do
    echo "which ${cmd}:"
    if command -v "${cmd}" >/dev/null 2>&1; then
      command -v "${cmd}"
    else
      echo "${cmd} not found"
    fi
    echo
  done

  echo "gevgen --help (first 40 lines):"
  if command -v gevgen >/dev/null 2>&1; then
    gevgen --help 2>&1 | sed -n '1,40p'
  else
    echo "gevgen not found"
  fi
  echo

  echo "gevgen_atmo --help (first 40 lines):"
  if command -v gevgen_atmo >/dev/null 2>&1; then
    gevgen_atmo --help 2>&1 | sed -n '1,40p'
  else
    echo "gevgen_atmo not found"
  fi
  echo
} | tee "${OUT_FILE}"
