#/bin/bash
export LC_ALL="en_US.UTF-8"

WORK_ROOT=workspace


echo ""
read -p "Set gpu nums(1-8)[default:4]: " GPU_NUM
if [ -z ${GPU_NUM} ]; then
    GPU_NUM=4
fi

echo ""
echo "dataset[default:imagenette]:"
i=1
for dirname in `ls configs`
do
    echo ${i}.${dirname}
    i=$(( i + 1 ))
done

read -p "select a dataset: " BASE_DIR
if [ -z ${BASE_DIR} ]; then
    BASE_DIR=1
fi

i=1
for dataset_name in `ls configs`
do
    if [ ${dataset_name} = ${BASE_DIR} -o ${i} = ${BASE_DIR} ]; then
        BASE_DIR=${dataset_name}
        ALIAS_NAME=${dataset_name}
        break
    fi
    i=$(( i + 1 ))
done

echo ""
echo "config file:"
i=1
for configfile in `ls configs/${BASE_DIR}`
do
    echo ${i}.${configfile}
    i=$(( i + 1 ))
done

read -p "select a config file: " CONFIG_FILE_NAME
if [ -z ${CONFIG_FILE_NAME} ]; then
    CONFIG_FILE_NAME=1
fi

i=1
for configfile in `ls configs/${BASE_DIR}`
do
    if [ ${configfile} = ${CONFIG_FILE_NAME} -o ${i} = ${CONFIG_FILE_NAME} ]; then
        CONFIG_FILE_NAME=${configfile}
        break
    fi
    i=$(( i + 1 ))
done

echo ""
read -p "Set log name: " LOG_NAME

DATETIME=`date +"%Y_%m_%d_%H" `
MODEL_TYPE=""

echo ""
echo "Start CMD List"
echo "*****************"
echo "1.train.sh(default)"
echo "2.sstrain.sh"
echo "4.test.sh"
echo "*****************"
read -p "select a shell: " selector_sh
if [ -z ${selector_sh} ]; then
    selector_sh=1
fi
case ${selector_sh} in
1|"train.sh")
    RUN_CMD=train.sh
    LOG_NAME=${ALIAS_NAME}_${LOG_NAME}
    TASK_FLAG=${LOG_NAME}_${DATETIME}
    ;;
2|"sstrain.sh")
    RUN_CMD=sstrain.sh

    read -p "Model type(e.g., resnet/vit)[default:resnet]:" MODEL_TYPE
    if [ -z ${MODEL_TYPE} ]; then
        MODEL_TYPE=resnet
    fi
    LOG_NAME=${ALIAS_NAME}_${LOG_NAME}
    TASK_FLAG=${LOG_NAME}_${DATETIME}
    ;;
3|"sstrain_elastic.sh")
    RUN_CMD=sstrain_elastic.sh
    LOG_NAME=${ALIAS_NAME}_${LOG_NAME}
    TASK_FLAG=${LOG_NAME}_${DATETIME}
    ;;
4|"test.sh")
    RUN_CMD=test.sh
    read -p "Test dataset(e.g., imagenet)[default:imagenet]:" TEST_DATASET
    if [ -z ${TEST_DATASET} ]; then
        TEST_DATASET=sph
    fi
    TASK_FLAG=${TEST_DATASET}-${LOG_NAME}
    ;;
5|"sleep.sh")
    RUN_CMD=sleep.sh
    TASK_FLAG=`date +"%Y_%m_%d_%H_%M_%S" `
    ;;
6|"resume_sstrain.sh")
    RUN_CMD=resume_sstrain.sh
    CONFIG_FILE_NAME=""
    TASK_FLAG=${LOG_NAME}
    ;;
*)
    echo "shell not exist"
    exit 1
    ;;
esac

DATETIME_SEC=`date +"%Y_%m_%d_%H_%M_%S" `

cp configs/${BASE_DIR}/${CONFIG_FILE_NAME} runningscripts/${DATETIME_SEC}_${CONFIG_FILE_NAME}

TASK_RUN_SCRIPT=${DATETIME_SEC}_${RUN_CMD}

echo "Generating the training shell."
echo "SRC: bin/${RUN_CMD} => DST: runningscripts/"${TASK_RUN_SCRIPT}
echo "Copying the config file."
echo "SRC: configs/${BASE_DIR}/${CONFIG_FILE_NAME} => DST: runningscripts/${DATETIME_SEC}_${CONFIG_FILE_NAME}"
echo "output dir => ${WORK_ROOT}/${TASK_FLAG}"

CONFIG_FILE_NAME=${DATETIME_SEC}_${CONFIG_FILE_NAME}

cat "bin/"${RUN_CMD}|sed \
-e "s/^TASK_WORK_DIR=.*/TASK_WORK_DIR=${TASK_FLAG}/" \
-e "s/^CONFIG_FILE_NAME=.*/CONFIG_FILE_NAME=runningscripts\/${CONFIG_FILE_NAME}/" \
-e "s/^multigpu=.*/multigpu=${GPU_NUM}/" \
-e "s/^basedir=.*/basedir=${BASE_DIR}/" \
-e "s/^modeltype=.*/modeltype=${MODEL_TYPE}/" \
> "runningscripts/"${TASK_RUN_SCRIPT}

sh "runningscripts/"${TASK_RUN_SCRIPT}
