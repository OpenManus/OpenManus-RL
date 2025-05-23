import os
import platform
import signal
import subprocess


def run_cmd(cmd_string, timeout=600):
    print("命令为：" + cmd_string)
    p = subprocess.Popen(
        cmd_string,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        shell=True,
        close_fds=True,
        start_new_session=True,
    )
    print("created")
    encoding = "utf-8"
    if platform.system() == "Windows":
        encoding = "gbk"

    try:
        print("trying")
        (msg, errs) = p.communicate(timeout=timeout)
        print("comed")
        ret_code = p.poll()
        print("polled")
        if ret_code:
            code = 1
            msg = "[Error]Called Error ： " + str(msg.decode(encoding))
        else:
            code = 0
            msg = str(msg.decode(encoding))
        print(ret_code)
    except subprocess.TimeoutExpired:
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGTERM)

        code = 1
        msg = (
            "[ERROR]Timeout Error : Command '"
            + cmd_string
            + "' timed out after "
            + str(timeout)
            + " seconds"
        )
    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)

    print("returning")

    return code, msg
