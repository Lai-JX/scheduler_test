
import itertools
import os
import subprocess
from multiprocessing import Process
import time

def run_job():
    command_to_run = 'docker exec -it muri /bin/bash -c "/root/nfs-share/mps_test/run_two_job.sh -g {} -f {}  -s {} -p {}"'.format(num_gpu, job[0], job[1], path1)
    print('run:',command_to_run)    
    output = run_command(command_to_run)
    print("run_job_exit\n")

def dcgmi(path):
    print(path)
    command = './dcgmi.sh {}'.format(path)
    try:
        # 执行命令并等待返回结果
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("dcgmi_exit\n")
        return result.stdout
    except Exception as e:
        # 如果命令执行失败，捕获异常并处理
        print(f"Command execution failed with error: {e}")
        return None




# 执行终端命令
def run_command(command):
    try:
        # 执行命令并等待返回结果
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，捕获异常并处理
        print(f"Command execution failed with error: {e}")
        return None



if __name__ == "__main__":
    jobs = ['resnet_18','shufflenet_v2_x1_0','vgg19','vgg16','dqn','a2c','bert','gpt2']
    combinations_lst = list(itertools.combinations(jobs, 2))
    combinations_lst = [('resnet_18', 'vgg19'), ('shufflenet_v2_x1_0', 'vgg19'), ('shufflenet_v2_x1_0', 'gpt2')]
    combinations_lst = [('vgg19','vgg16'),('resnet_18', 'vgg19')]
    combinations_lst = [('bert','gpt2')]
    combinations_lst = [('vgg16','dqn'),('bert','gpt2'),('dqn','a2c')]
    combinations_lst = [('vgg19','gpt2')]
    print(combinations_lst)
    count = 0
    # for num_gpu in [1,2,4,8]:
    for num_gpu in [8]:
        
        dir1 = '/home/jxlai/nfs-share/mps_test/IterTime/two_job1_mps/gpu{}/'.format(num_gpu)
        dir2 = '/home/jxlai/nfs-share/mps_test/result/two_job1_mps/gpu{}/'.format(num_gpu)
        os.system('mkdir -p {}; mkdir -p {};'.format(dir1, dir2))
        for job in combinations_lst:
            # 启动job
            path1 = (dir1+job[0]+'--'+job[1]).replace('/home/jxlai', '/root')
            p1 = Process(target=run_job)
            p1.start()

            time.sleep(10)
            # 收集指标数据
            p2 = Process(target=dcgmi, args=[dir2+job[0]+'--'+job[1]])
            p2.start()

            p1.join()
            p2.join()
            count += 1
        if num_gpu==1:
            run_command('dcgmi group -g 2 -a 1')
        if num_gpu==2:
            run_command('dcgmi group -g 2 -a 2')
            run_command('dcgmi group -g 2 -a 3')
        if num_gpu==4:
            run_command('dcgmi group -g 2 -a 4')
            run_command('dcgmi group -g 2 -a 5')
            run_command('dcgmi group -g 2 -a 6')
            run_command('dcgmi group -g 2 -a 7')
    print(count)

    # 重新启动时，记得：修改目录，修改group
        
