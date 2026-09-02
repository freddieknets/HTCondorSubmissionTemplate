#!/bin/bash

ENVNAME=potato_please_work
environments=(fluka)

# XSUITEPATH=''   # Install xsuite from PyPI
XSUITEPATH=/afs/cern.ch/work/l/lbertolo/public/soft/xsuite/

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
    source ${STUDYPATH}/submission_scripts/environment.sh "${environments[@]}"
    echo "Creating Xsuite environment..."
    python -m venv --system-site-packages build_venv
    source build_venv/bin/activate
    echo "Installing packages..."
    python -m pip install -U pip setuptools wheel distutils setuptools-scm[toml]
    if [ "${XSUITEPATH}" == '']
    then
        python -m pip install xsuite
    else
        for pkg in xobjects xdeps xtrack xpart xfields xcoll; do
            python -m pip install --upgrade --force-reinstall --no-deps ${XSUITEPATH}$pkg
        done
        #python -m pip install --upgrade ${XSUITEPATH}xobjects
        #python -m pip install --upgrade ${XSUITEPATH}xdeps
        #python -m pip install --upgrade ${XSUITEPATH}xtrack
        #python -m pip install --upgrade ${XSUITEPATH}xpart
        #python -m pip install --upgrade ${XSUITEPATH}xfields
        #python -m pip install --upgrade ${XSUITEPATH}xcoll
        # Do not get the local version of xsuite nor wheels to avoid kernel version conflicts
        python -m pip install --upgrade --force-reinstall xsuite --no-deps --no-binary=xsuite
    fi
    # How to automatise for different environments?
    if [[ ${environments[*]} =~ (^|[[:space:]])"fluka"($|[[:space:]]) ]]
    then
        echo "Initializing FLUKA..."
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
if [ -f ${SPOOLPATH}files_${STUDYNAME}.tar.gz ]
then
    rm ${SPOOLPATH}files_${STUDYNAME}.tar.gz
fi
tar -C . -czf ${SPOOLPATH}files_${STUDYNAME}.tar.gz scripts data -C ${ENVPATH} $envfile -C ${STUDYPATH}/submission_scripts environment.sh
