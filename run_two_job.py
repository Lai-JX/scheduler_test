
import itertools
import os
import subprocess
from multiprocessing import Process
import time

def run_job(num_gpu, job0, bs0, job1, bs1, path):
    if job1 is None:
        command_to_run = '/workspace/nfs-share/mps_test/run_two_job.sh -g {} -f {} -x {} -p {}'.format(num_gpu, job0, bs0, path)  
    else:
        command_to_run = '/workspace/nfs-share/mps_test/run_two_job.sh -g {} -f {} -x {} -s {}  -y {} -p {}'.format(num_gpu, job0, bs0, job1, bs1, path)  
    print('run:',command_to_run)    
    output = run_command(command_to_run)
    print("run_job_exit\n")
    print("------------------------------------------------------------------------------")

def dcgmi(path):
    print(path)
    command = './dcgmi.sh {}'.format(path)
    try:
        # 执行命令并等待返回结果
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("dcgmi_exit")
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
    # combinations_lst = [('resnet_18', 'vgg19'), ('shufflenet_v2_x1_0', 'vgg19'), ('shufflenet_v2_x1_0', 'gpt2')]
    # combinations_lst = [('vgg19','vgg16'),('resnet_18', 'vgg19')]
    # combinations_lst = [('bert','gpt2')]
    # combinations_lst = [('vgg16','dqn'),('bert','gpt2'),('dqn','a2c')]
    # combinations_lst = [('vgg19','gpt2')]
    combinations_lst = ['shufflenet_v2_x1_0','vgg19','vgg16','dqn','a2c','bert','gpt2']
    print(combinations_lst)
    count = 0
    for num_gpu in [1,2,4,8]:
    # for num_gpu in [8]:
        
        dir1_1 = '/workspace/nfs-share/mps_test/IterTime/single/gpu{}/'.format(num_gpu)
        dir2_1 = '/workspace/nfs-share/mps_test/result/single/gpu{}/'.format(num_gpu)
        os.system('mkdir -p {}; mkdir -p {};'.format(dir1_1, dir2_1))
        for job in combinations_lst:
            if isinstance(job, str):  
                dir1_2, dir2_2 = dir1_1+job+'/', dir2_1+job+'/'
            else:
                dir1_2, dir2_2 = dir1_1+job[0]+'--'+job[1]+'/', dir2_1+job[0]+'--'+job[1]+'/'
            os.system('mkdir -p {}; mkdir -p {};'.format(dir1_2, dir2_2))
            for bs in [32, 128, 512, 2048]:
                # 启动job
                if isinstance(job, str):                                            # 单一job 
                    p1 = Process(target=run_job, args=[num_gpu, job, bs, None, None, dir1_2+str(bs)])
                else:                                                               # 2 jobs
                    p1 = Process(target=run_job, args=[num_gpu, job[0], bs, job[1], bs, dir1_2+str(bs)])
                p1.start()

                time.sleep(10)
                # 收集指标数据
                p2 = Process(target=dcgmi, args=[dir2_2+str(bs)])
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
        
