#!/bin/bash
export PYTHON=python3
PROJ_PATH=.
cd ${PROJ_PATH}
WORK_ROOT=./result

runModel(){
    CONFIG_FILE=$1
    WORK_DIR=${WORK_ROOT}/$2
    rm -rf ${WORK_DIR}
    if [ ${multigpu} -gt 1 ];then
        ${PYTHON} -m torch.distributed.launch --nproc_per_node=${multigpu} --master_addr 127.0.0.1 --master_port ${PORT} \
            linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR} --multigpu ${multigpu} --syncbn ${syncbn} --mixed ${mixed}
    else
        ${PYTHON} linear_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR} --mixed ${mixed}
    fi
    ${PYTHON} tools/get_max.py --workdir ${WORK_DIR}
}

DATETIME=`date +"%Y_%m_%d_%H" `
multigpu=4
PORT=28500
syncbn=1
mixed=1

TASK_WORK_DIR=test_${DATETIME}
CONFIG_FILE_NAME=tmpfile.py

runModel ${CONFIG_FILE_NAME} ${TASK_WORK_DIR}
