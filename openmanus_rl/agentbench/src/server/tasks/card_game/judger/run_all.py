import sys
import os, shutil
import signal
import subprocess
import platform
import time
import json
from .cal_metric import calculate

def run_cmd(cmd_string, timeout=600):
    print("命令为：" + cmd_string)
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                         start_new_session=True)
 
    format = 'utf-8'
    if platform.system() == "Windows":
        format = 'gbk'
 
    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error ： " + str(msg.decode(format))
        else:
            code = 0
            msg = str(msg.decode(format))
    except subprocess.TimeoutExpired:
        # 注意：不能只使用p.kill和p.terminate，无法杀干净所有的子进程，需要使用os.killpg
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)
 
        # 注意：如果开启下面这两行的话，会等到执行完成才报超时错误，但是可以输出执行结果
        # (outs, errs) = p.communicate()
        # print(outs.decode('utf-8'))
 
        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)
 
    return code, msg

base_model = [
    'baseline1',
    'baseline2',
]

root_dir = '..'

location = {
    'ai': f'python+{root_dir}/AI_SDK/Python/main.py+en+%s+%d+%d+%s',
    'baseline1': f'python+{root_dir}/AI_SDK/Python/basline1.py+%d+%d+%s',
    'baseline2': f'python+{root_dir}/AI_SDK/Python/basline2.py+%d+%d+%s',
}

result_dir = f'{root_dir}/result'

def batch_test(ai, base, stage, test_times):
    
    for i in range(test_times):
        stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(int(time.time())))
        save_dir = f'{result_dir}/{stage}_{ai.name}_{base}/{stamp}'
        os.makedirs(save_dir)

        cmd = f'python judger.py {root_dir}/logic/bin/main3 %s %s config {save_dir}/replay.json'
        msg = run_cmd(cmd % (location["ai"] % (ai, stage, 0, save_dir), location[base] % (stage, 1, save_dir)))[1]
        time.sleep(1)
        print(msg)
        
        meta = {'ai1': ai.name, 'ai2': base}
        
        if "\"0\" : 0" in msg:
            meta['winner'] = '1'
        else:
            meta['winner'] = '0'
                
        with open(save_dir + '/meta.json', 'w') as f:
            f.write(json.dumps(meta))

def get_llm(ai, test_time=50):
    output_path = "result.csv"
    ret = []
    for stage in [2, 1]:
        # run and eval model
        for base in base_model:
            print(stage, ai, base)
            batch_test(ai, base, stage, test_time)
            dict1 = calculate("../result/%d_%s_%s" % (stage, ai.name, base), 0)
            ret.append({
                'stage': stage,
                'ai': ai.name,
                'base_model': base,  
                'first': True,
                'result': dict1,
            })
            
            dict2 = calculate("../result/%d_%s_%s" % (stage, base, ai.name), 1)
            ret.append({
                'stage': stage,
                'ai': ai.name,
                'base_model': base,  
                'first': False,
                'result': dict2,
            })
    return ret