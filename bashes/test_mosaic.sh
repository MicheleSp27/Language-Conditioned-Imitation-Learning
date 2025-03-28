#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3

#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

export MUJOCO_PY_MUJOCO_PATH="/user/frosa/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

echo $1
TASK_NAME="$1"
NUM_WORKERS=20
GPU_ID=1

BASE_PATH=/raid/home/frosa_Loc/Language-Conditioned-Imitation-Learning
CKP_FOLDER=/user/frosa/RT-1-Checkpoint
if [ "$TASK_NAME" == 'pick_place' ]; then
    PROJECT_NAME=rt-1_pretraining_simulated
    BATCH=32 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 1; do #81000 89100; do
            for TASK in pick_place; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --debug --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --debug --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'nut_assembly' ]; then
    PROJECT_NAME=1Task-nut_assembly-MOSAIC-State_pos_gripper
    BATCH=27 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 164032; do #81000 89100; do
            for TASK in nut_assembly; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'button' ]; then
    PROJECT_NAME=1Task-press_button-MOSAIC-State_pos_gripper
    BATCH=18 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in -1; do #81000 89100; do
            for TASK in button; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log #--save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'stack_block' ]; then
    PROJECT_NAME=1Task-stack_block-MOSAIC-State_pos_gripper
    BATCH=18 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in -1; do
            for TASK in stack_block; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log #--save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'multi' ]; then
    echo "Multi Task"
    PROJECT_NAME=4Task-MOSAIC-State_pos_gripper
    BATCH=74
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

    for MODEL in ${MODEL_PATH}; do
        for S in 253890; do
            for TASK in pick_place nut_assembly stack_block button; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done

fi
