#!/bin/bash
export LC_ALL="en_US.UTF-8"
export PYTHON=python3
PROJ_PATH=.
cd ${PROJ_PATH}
WORK_ROOT=./result

DATETIME=`date +"%Y_%m_%d_%H" `

ssModel(){
    CONFIG_FILE=$1
    WORK_DIR=${WORK_ROOT}/$2
    rm -rf ${WORK_DIR}

    if [ ${multigpu} -gt 1 ];then
        ${PYTHON} -m torch.distributed.launch --nproc_per_node=${multigpu} --master_addr 127.0.0.1 --master_port ${PORT} \
            ss_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR} --multigpu ${multigpu} --syncbn ${syncbn} --mixed ${mixed} --mode ${model}
    else
        ${PYTHON} ss_train.py --config ${CONFIG_FILE} --workdir ${WORK_DIR} --mixed ${mixed} --mode ${model}
    fi
}
transCKPT(){
    WORK_DIR=${WORK_ROOT}/$1
    LAST_CKPT=${WORK_DIR}/epoch_end.pth
    BACKBONE_CKPT=${WORK_DIR}/backbone.pth
    ${PYTHON} tools/process_ckpt.py --checkpoint ${LAST_CKPT} --output ${BACKBONE_CKPT}
}
linearModel(){
    LINEAR_CFG_FILE=configs/${basedir}/${EVAL_FILE}
    LINEAR_WORK_DIR=${WORK_ROOT}/$1/linear
    BACKBONE_CKPT=$2

    rm -rf ${LINEAR_WORK_DIR}

    if [ ${multigpu} -gt 1 ];then
        ${PYTHON} -m torch.distributed.launch --nproc_per_node=${multigpu} --master_addr 127.0.0.1 --master_port ${PORT} \
            linear_train.py --config ${LINEAR_CFG_FILE} --workdir ${LINEAR_WORK_DIR} --pretrained ${BACKBONE_CKPT} --multigpu ${multigpu} --syncbn ${syncbn} --mixed ${mixed} --basedir ${basedir}
    else
        ${PYTHON} linear_train.py --config ${LINEAR_CFG_FILE} --workdir ${LINEAR_WORK_DIR} --pretrained ${BACKBONE_CKPT} --mixed ${mixed} --basedir ${basedir}
    fi

    ${PYTHON} tools/get_max.py --workdir ${LINEAR_WORK_DIR}
}

runModel(){
    ssModel $1 $2
    transCKPT $2
    BACKBONE_CKPT=${WORK_ROOT}/$2/backbone.pth
    linearModel $2 ${BACKBONE_CKPT}
}

multigpu=4
basedir=imagenette
modeltype=resnet

if [[ $modeltype = "resnet" ]]
then
    EVAL_FILE="linear_probe.py"
else
    EVAL_FILE="linear_probe_vit.py"
fi

PORT=28500
syncbn=1
mixed=1
model=SSModel

TASK_WORK_DIR=test_${DATETIME}
CONFIG_FILE_NAME=tmpfile.py

runModel ${CONFIG_FILE_NAME} ${TASK_WORK_DIR}
