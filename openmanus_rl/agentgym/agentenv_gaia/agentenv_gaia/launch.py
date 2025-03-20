import argparse
import os

from agentenv_gaia.server import launch as server_launch


def main():
    """
    GAIA环境服务器启动脚本
    """
    parser = argparse.ArgumentParser(description="启动GAIA环境服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--data-dir", type=str, default="data/", help="数据目录路径")

    args = parser.parse_args()

    # 确保数据目录存在
    os.makedirs(os.path.join(args.data_dir, "gaia"), exist_ok=True)

    print(f"启动GAIA环境服务器，监听地址: {args.host}:{args.port}")
    print(f"数据目录: {args.data_dir}")

    # 启动服务器
    server_launch(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
