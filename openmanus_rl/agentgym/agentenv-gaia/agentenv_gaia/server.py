import threading
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentenv_gaia.enviroment import GaiaEnv


# 请求模型定义
class CreateEnvRequest(BaseModel):
    data_dir: Optional[str] = "data/"
    level: Optional[str] = "level1"
    dataset: Optional[str] = "validation"
    tool_list: Optional[List[str]] = None


class StepRequest(BaseModel):
    env_id: str
    action: str


class ResetRequest(BaseModel):
    env_id: str
    idx: int = 0


# 环境服务器类
class GaiaEnvServer:
    def __init__(self, max_envs=100):
        self.env_instances = {}  # 存储环境实例
        self.env_locks = {}  # 环境锁，用于并发控制
        self.max_envs = max_envs

    def create_env(self, params: CreateEnvRequest):
        """创建新的环境实例"""
        if len(self.env_instances) >= self.max_envs:
            raise HTTPException(status_code=503, detail="达到最大环境实例数量限制")

        env_id = str(uuid.uuid4())
        self.env_instances[env_id] = GaiaEnv(
            data_dir=params.data_dir,
            level=params.level,
            dataset=params.dataset,
            tool_list=params.tool_list,
        )
        self.env_locks[env_id] = threading.Lock()

        return {"env_id": env_id, "status": "created"}

    def get_observation(self, env_id: str):
        """获取环境观察"""
        self._check_env_id(env_id)

        with self.env_locks[env_id]:
            return self.env_instances[env_id].get_observation()

    def get_available_actions(self, env_id: str):
        """获取可用动作"""
        self._check_env_id(env_id)

        with self.env_locks[env_id]:
            return self.env_instances[env_id].get_available_actions()

    def step(self, env_id: str, action: str):
        """执行动作"""
        self._check_env_id(env_id)

        with self.env_locks[env_id]:
            observation, reward, done, truncated, info = self.env_instances[
                env_id
            ].step(action)
            return {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
            }

    def reset(self, env_id: str, idx: int = 0):
        """重置环境"""
        self._check_env_id(env_id)

        with self.env_locks[env_id]:
            observation, info = self.env_instances[env_id].reset(options={"idx": idx})
            return {"observation": observation, "info": info}

    def _check_env_id(self, env_id: str):
        """检查环境ID是否存在"""
        if env_id not in self.env_instances:
            raise HTTPException(status_code=404, detail=f"环境实例 {env_id} 不存在")


# 创建服务器实例
server = GaiaEnvServer()

# 创建FastAPI应用
app = FastAPI(title="GAIA Environment Server")


@app.get("/")
def hello():
    return {"message": "欢迎使用GAIA环境服务器"}


@app.post("/createEnv")
def create_env(params: CreateEnvRequest = CreateEnvRequest()):
    return server.create_env(params)


@app.get("/observation")
def get_observation(env_id: str):
    return server.get_observation(env_id)


@app.get("/available_actions")
def get_available_actions(env_id: str):
    return server.get_available_actions(env_id)


@app.post("/step")
def step(request: StepRequest):
    return server.step(request.env_id, request.action)


@app.post("/reset")
def reset(request: ResetRequest):
    return server.reset(request.env_id, request.idx)


# 启动函数
def launch(host="0.0.0.0", port=8000):
    """启动GAIA环境服务器

    Args:
        host: 服务器主机地址
        port: 服务器端口
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    launch()
