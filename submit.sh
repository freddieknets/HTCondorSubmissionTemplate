#!/bin/bash

STUDYNAME=ExampleStudy
JOBSFILE=example.jobs.yaml

ENVNAME=0.45.15_geant4
environments=(geant4)

XSUITEPATH=/eos/home-f/fvanderv/pythondev/
AFSSUBMISSIONPATH=/afs/cern.ch/work/f/fvanderv/SimulationSubmissions/

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
    source submission_scripts/environment.sh "${environments[@]}"
    echo "Creating Xsuite environment..."
    python -m venv --system-site-packages build_venv
    source build_venv/bin/activate
    echo "Installing packages..."
    pip install -U pip setuptools wheel distutils setuptools-scm[toml]
    pip install ${XSUITEPATH}xobjects
    pip install ${XSUITEPATH}xdeps
    pip install ${XSUITEPATH}xtrack
    pip install ${XSUITEPATH}xpart
    pip install ${XSUITEPATH}xfields
    pip install ${XSUITEPATH}xcoll
    pip install --no-dependencies ${XSUITEPATH}xsuite
    echo "Prebuilding xsuite kernels..."
    xsuite-prebuild r
    # How to automatise for different environments?
    if [[ ${environments[*]} =~ (^|[[:space:]])"fluka"($|[[:space:]]) ]]
    then
        echo "Initializing FLUKA..."
        python ../scripts/fluka_init_eos.py
    fi
    if [[ ${environments[*]} =~ (^|[[:space:]])"geant4"($|[[:space:]]) ]]
    then
        echo "Initializing Geant4..."
        python ../scripts/geant4_init.py
    fi
    echo "Packing environment..."
    pip install venv-pack
    venv-pack -p build_venv -o ${ENVPATH}$envfile
    deactivate
    rm -r build_venv
    echo "Environment created."
    echo
    cd $STUDYPATH
fi


# Spool the necessary files
echo "Spooling files..."
if [ -f ${SPOOLPATH}files_${STUDYNAME}.tar.gz ]
then
    rm ${SPOOLPATH}files_${STUDYNAME}.tar.gz
fi
tar -C . -czf spool/files_${STUDYNAME}.tar.gz scripts data -C envs $envfile -C ../submission_scripts environment.sh
echo


# Prepare submission directory and submit the job
DIR=${AFSSUBMISSIONPATH}${STUDYNAME}/
mkdir -p $DIR
if [ -L $STUDYNAME ] || [ -f $STUDYNAME ]
then
    rm $STUDYNAME
fi
ln -fns $DIR $STUDYNAME

cp submission_scripts/job.sh $DIR
cp submission_scripts/submission.sub $DIR
echo "Generating job list..."
python submission_scripts/generate_jobs.py --spec $JOBSFILE --out ${DIR}jobs.list --preview
echo "Job list generated."
echo

cd $DIR
if [ ${#environments[@]} -gt 0 ]
then
    ENV_LIST="${environments[*]}"
    condor_submit NAME="$STUDYNAME" PATH="$STUDYPATH" ENV_LIST="$ENV_LIST" submission.sub
else
    condor_submit NAME="$STUDYNAME" PATH="$STUDYPATH" submission.sub
fi
