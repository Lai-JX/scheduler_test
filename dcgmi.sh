# echo
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

dcgmi dmon -g 2 -e 1002,1003,1005,1004 >  $1 &
sleep 140s
kill_all "dcgmi"
exit 0 