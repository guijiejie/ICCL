#!/bin/bash
export LC_ALL="en_US.UTF-8"
export PYTHON=python3
PROJ_PATH=.
cd ${PROJ_PATH}
WORK_ROOT=./result

DATETIME=`date +"%Y_%m_%d_%H" `

transCKPT(){
    WORK_DIR=${WORK_ROOT}/$1
    LAST_CKPT=${WORK_DIR}/epoch_end.pth
    BACKBONE_CKPT=${WORK_DIR}/backbone.pth
    ${PYTHON} tools/process_ckpt.py --checkpoint ${LAST_CKPT} --output ${BACKBONE_CKPT}
}
linearModel(){
    LINEAR_CFG_FILE=${CONFIG_FILE_NAME}
    LINEAR_WORK_DIR=${WORK_ROOT}/$1/linear$3
    BACKBONE_CKPT=$2

    rm -rf ${LINEAR_WORK_DIR}

    if [ ${multigpu} -gt 1 ];then
        ${PYTHON} -m torch.distributed.launch --nproc_per_node=${multigpu} --master_addr 127.0.0.1 --master_port ${PORT} \
            linear_train.py --config ${LINEAR_CFG_FILE} --workdir ${LINEAR_WORK_DIR} --pretrained ${BACKBONE_CKPT} --multigpu ${multigpu} --syncbn ${syncbn} --mixed ${mixed}
    else
        ${PYTHON} linear_train.py --config ${LINEAR_CFG_FILE} --workdir ${LINEAR_WORK_DIR} --pretrained ${BACKBONE_CKPT} --mixed ${mixed}
    fi

    ${PYTHON} tools/get_max.py --workdir ${LINEAR_WORK_DIR}
}
linclsModel(){
    transCKPT $1
    BACKBONE_CKPT=${WORK_ROOT}/$1/backbone.pth
    linearModel $1 ${BACKBONE_CKPT} ${TEST_DATASET}
}

multigpu=4

PORT=28500
syncbn=1
mixed=1

TASK_WORK_DIR=imagenet-test
TEST_DATASET=${TASK_WORK_DIR%%-*}
REAL_TASK_WORK_DIR=${TASK_WORK_DIR##*-}
echo ${TEST_DATASET}
echo ${REAL_TASK_WORK_DIR}
CONFIG_FILE_NAME=configs/imagenet/linear_probe.py

linclsModel ${REAL_TASK_WORK_DIR}
