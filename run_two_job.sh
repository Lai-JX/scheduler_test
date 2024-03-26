#! /bin/bash

cmd=" --allow-run-as-root --oversubscribe --prefix /root/share/openmpi -bind-to none -map-by slot -mca btl_tcp_if_include 10.249.46.12/21  -x LD_LIBRARY_PATH -x PATH  -mca pml ob1 -mca btl ^openib /root/anaconda3/envs/muri/bin/python3 /root/nfs-share/mps_test/main_util_single.py "

resnet_18="--model0 resnet18 --batch-size0 128 --train-dir0 /root/datasets/ILSVRC2012/train/ --num-workers0 0 --prefetch-factor0 2 --iters0 5280 --job-id0 0  --this-dir /root/nfs-share/mps_test"
shufflenet_v2_x1_0="--model0 shufflenet_v2_x1_0 --batch-size0 128 --train-dir0 /root/datasets/ILSVRC2012/train/ --num-workers0 0 --prefetch-factor0 2 --iters0 5020 --job-id0 0  --this-dir /root/nfs-share/mps_test"
vgg19="--model0 vgg19 --batch-size0 16 --train-dir0 /root/datasets/ILSVRC2012/train/ --num-workers0 0 --prefetch-factor0 2 --iters0 2000 --job-id0 0  --this-dir /root/nfs-share/mps_test"
vgg16="--model0 vgg16 --batch-size0 16 --train-dir0 /root/datasets/ILSVRC2012/train/ --num-workers0 0 --prefetch-factor0 2 --iters0 4000 --job-id0 0  --this-dir /root/nfs-share/mps_test"
dqn="--model0 dqn --batch-size0 128 --train-dir0 ./ --num-workers0 0 --prefetch-factor0 2 --iters0 5000 --job-id0 0  --this-dir /root/nfs-share/mps_test"
a2c="--model0 a2c --batch-size0 64 --train-dir0 ./ --num-workers0 0 --prefetch-factor0 2 --iters0 1200 --job-id0 0  --this-dir /root/nfs-share/mps_test"
bert="--model0 bert --batch-size0 4 --train-dir0 /root/nfs-share/Muri_exp/workloads/wikitext-2-raw/wiki.train.raw --num-workers0 0 --prefetch-factor0 2 --iters0 3000 --job-id0 0  --this-dir /root/nfs-share/mps_test"
gpt2="--model0 gpt2 --batch-size0 4 --train-dir0 /root/nfs-share/Muri_exp/workloads/wikitext-2-raw/wiki.train.raw --num-workers0 0 --prefetch-factor0 2 --iters0 3000 --job-id0 0  --this-dir /root/nfs-share/mps_test"
kill_all()
{
    # ID=`ps -ef | grep "$1" | grep -v "$0" | grep -v "grep" | awk '{print $2}'`  # -v表示反过滤，awk表示按空格或tab键拆分，{print $2}表示打印第二个（这里对应进程号）
    ID=`ps | grep "$1" | grep -v "$0" | grep -v "grep" | awk '{print $1}'`  # -v表示反过滤，awk表示按空格或tab键拆分，{print $2}表示打印第二个（这里对应进程号）
    for id in $ID 
    do  
    echo "killed $id"  
    kill -9 $id  
    # echo "killed $id"  
    done
}

# 1. 获取参数
num_gpu=1
job1_str=''
job2_str=''
job1=''
job2=''
path=''
while getopts "g:f:s:p:h" arg #选项后面的冒号表示该选项需要参数
do
        case $arg in
             g)
                num_gpu=$OPTARG
                ;;
             f)
                job1_str=$OPTARG
                job1=`eval echo '$'"$job1_str"`
                # echo $job1
                ;;
             s)
                job2_str=$OPTARG
                job2=`eval echo '$'"$job2_str"`
                # echo $job2
                ;;
             p)
                path=$OPTARG
                # echo $job2
                ;;
             h)
                echo "-g                The number of gpu."
                echo "-f                First job"
                echo "-s                Second job"
                echo "-p                IterTime output path"
                exit
                ;;
             ?)  #当有不认识的选项的时候arg为?
             
            echo "unkonw argument"
        exit 1
        ;;
        esac
done

cmd1=$num_gpu$cmd$job1
cmd2=$num_gpu$cmd$job2
# echo $cmd1
# echo $cmd2

/root/share/openmpi/bin/mpirun -n $cmd1 > $path-$job1_str &
/root/share/openmpi/bin/mpirun -n $cmd2 > $path-$job2_str &

sleep 155s
kill_all "python"
exit 0
