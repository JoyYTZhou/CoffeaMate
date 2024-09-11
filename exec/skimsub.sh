# ==============================================================================
# Updated on: June 3, 2024
# Used to: create dynamic job submissions for different datasets
# ==============================================================================

DISABLE_SUBMISSION=false

while getopts ":d" opt; do
  case ${opt} in
    d )
      DISABLE_SUBMISSION=true
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument" 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

DYNACONF_ENV=$1
PROCESS=$2

cd ..
source scripts/venv.sh $DYNACONF_ENV
cd exec

JOB_DIRNAME=$(python3 -c 'from src.analysis.spawnjobs import rs; print(rs.JOB_DIRNAME)')
rm -rf ${JOB_DIRNAME}/*.json
python3 genjobs.py

if [ "$PROCESS" = "ALL" ]; then
    FILENAME="${JOB_DIRNAME}/*.json"
else 
    FILENAME="${JOB_DIRNAME}/${PROCESS}_*.json"
fi


\cp -f hhbbtt.sub runtime/${DYNACONF_ENV}_${PROCESS}.sub

cat << EOF >> runtime/${DYNACONF_ENV}_${PROCESS}.sub
DYNACONF = ${DYNACONF_ENV}
JOB_DIRNAME = ${JOB_DIRNAME}
queue FILENAME matching files ${FILENAME}
EOF

if [ "$DISABLE_SUBMISSION" = false ]; then
    condor_submit runtime/${DYNACONF_ENV}_${PROCESS}.sub
else
    echo "Submission disabled for process: $PROCESS"
fi


