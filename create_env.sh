#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: ${0##*/} [OPTIONS]

Creates and packs a venv with Xsuite (and optional simulation environments)
for shipping to the Condor cluster.

Options:
  -n, --name NAME          Environment name (required)
  -e, --environments LIST  Space- or comma-separated extra environments to
                           initialise, e.g. "fluka geant4". Default: none.
  -x, --xsuite-path PATH   Local Xsuite source dir to install from. If omitted,
                           Xsuite is installed from PyPI.
  -h, --help               Show this help and exit.

Examples:
  ${0##*/} -n my_study
  ${0##*/} -n my_study -e fluka
  ${0##*/} -n my_study -e "fluka geant4" -x /afs/cern.ch/work/l/lbertolo/public/soft/xsuite/
EOF
}

# Defaults
ENVNAME=""
XSUITEPATH=""          # empty => install from PyPI
environments=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)
            ENVNAME="$2"; shift 2 ;;
        -e|--environments)
            # accept "fluka,geant4" or "fluka geant4"
            IFS=', ' read -r -a environments <<< "$2"; shift 2 ;;
        -x|--xsuite-path)
            XSUITEPATH="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; break ;;
        -*)
            echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
        *)
            echo "Unexpected argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "$ENVNAME" ]]; then
    echo "Error: --name is required." >&2
    usage >&2
    exit 1
fi

# Validate the xsuite path if one was given
if [[ -n "$XSUITEPATH" && ! -d "$XSUITEPATH" ]]; then
    echo "Error: --xsuite-path '$XSUITEPATH' is not a directory." >&2
    exit 1
fi

STUDYPATH=$(pwd -P)
ENVPATH=${STUDYPATH}/envs/
mkdir -p ${ENVPATH}
SPOOLPATH=${STUDYPATH}/spool/
mkdir -p $SPOOLPATH

# Get or create the xsuite environment
envfile=xsuite_env_${ENVNAME}.tar.gz
if [ ! -f ${ENVPATH}$envfile ]
then
    cd $SPOOLPATH
    echo "Sourcing environment..."
    set +u
    source ${STUDYPATH}/submission_scripts/environment.sh "${environments[@]}"
    set -u
    unset PYTHONPATH # otherwise CVMFs python paths can cause conflicts with packages below
    echo "Creating Xsuite environment..."
    python -m venv --system-site-packages build_venv
    source build_venv/bin/activate
    echo "Installing packages..."
    # setuptools must be smaller than 75 to be compatible with the numpy version on LCG
    python -m pip install -U pip 'setuptools<75' wheel setuptools-scm[toml]
    if [ "${XSUITEPATH}" == '']
    then
        python -m pip install xsuite
    else
        for pkg in xobjects xdeps xtrack xpart xfields xcoll; do
            python -m pip install --upgrade --force-reinstall --no-deps ${XSUITEPATH}$pkg
        done
        # Do not get the local version of xsuite nor wheels to avoid kernel version conflicts
        python -m pip install --upgrade --force-reinstall xsuite --no-deps --no-binary=xsuite
    fi
    # How to automatise for different environments?
    if [[ ${environments[*]} =~ (^|[[:space:]])"fluka"($|[[:space:]]) ]]
    then
        echo "Initializing FLUKA..."
        python -m pip install --upgrade --force-reinstall 'meson>=1.4.0'
        python ${STUDYPATH}/submission_scripts/fluka_init_eos.py
    fi
    if [[ ${environments[*]} =~ (^|[[:space:]])"geant4"($|[[:space:]]) ]]
    then
        echo "Initializing Geant4..."
        python ${STUDYPATH}/submission_scripts/geant4_init.py
    fi
    echo "Packing environment..."
    pip install venv-pack
    venv-pack -p build_venv -o ${ENVPATH}$envfile
    deactivate
    rm -r build_venv
    echo "Environment created."
    echo
fi
cd $STUDYPATH


# Spool the necessary files
echo "Spooling files..."
if [ -f ${SPOOLPATH}files.tar.gz ]
then
    rm ${SPOOLPATH}files.tar.gz
fi
tar -C . -czf ${SPOOLPATH}files.tar.gz scripts data -C ${ENVPATH} $envfile -C ${STUDYPATH}/submission_scripts environment.sh
