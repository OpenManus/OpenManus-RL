import argparse
import hashlib
import json
import os

import docker
import yaml

client = docker.from_env()


def get_file_hash(file_path):
    """Function to get hash of a file"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()


def build_images(force=False):
    """Function to build docker images"""
    dockerfile_directory = os.path.join(args.root, CONFIG["docker_config"]["directory"])
    for filename in os.listdir(dockerfile_directory):
        if filename not in CONFIG["data_config"]["ignore"]:
            image_name = f'{CONFIG["docker_config"]["localhost"]}/{filename}'
            dockerfile_path = os.path.join(dockerfile_directory, filename)
            try:
                image = client.images.get(image_name)
                if not force:
                    # Check if the dockerfile has changed
                    if image.labels.get('file_hash') != get_file_hash(dockerfile_path):
                        # If dockerfile has changed, rebuild image
                        print(f'Rebuilding image: {image_name}')
                        client.images.build(path=dockerfile_directory, dockerfile=filename, tag=image_name, labels={'file_hash': get_file_hash(dockerfile_path)})
                    else:
                        print(f'Image: {image_name} up to date.')
                else:
                    print(f'Rebuilding image: {image_name}')
                    client.images.build(path=dockerfile_directory, dockerfile=filename, tag=image_name, labels={'file_hash': get_file_hash(dockerfile_path)})
            except docker.errors.ImageNotFound:
                # If image does not exist, build it
                print(f'Building image: {image_name}')
                client.images.build(path=dockerfile_directory, dockerfile=filename, tag=image_name, labels={'file_hash': get_file_hash(dockerfile_path)})


def clean_images():
    """Function to clean docker images"""
    dockerfile_directory = os.path.join(args.root, CONFIG["docker_config"]["directory"])
    for filename in os.listdir(dockerfile_directory):
        if filename not in CONFIG["data_config"]["ignore"]:
            image_name = f'{CONFIG["docker_config"]["localhost"]}/{filename}'
            try:
                image = client.images.get(image_name)
                client.images.remove(image.id)
                print(f'Removed image: {image_name}')
            except docker.errors.ImageNotFound:
                print(f'Image not found: {image_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manage Docker images.')
    parser.add_argument('command', choices=['build', 'clean'], help='The command to execute.')
    parser.add_argument('-c', '--config', default='config.json', help='The config file to use.')
    parser.add_argument('-f', '--force', action='store_true', help='Force rebuild of images.')
    parser.add_argument('-r', '--root', default='.', help='The root directory to use.')

    args = parser.parse_args()

    if args.config.endswith('.yaml'):
        with open(args.config, "r") as f:
            CONFIG = yaml.safe_load(f)["parameters"]
    elif args.config.endswith('.json'):
        with open(args.config, "r") as f:
            CONFIG = json.load(f)["parameters"]
    else:
        raise ValueError(f"Unknown config file type: {args.config}")

    if args.command == 'build':
        build_images(force=args.force)
    elif args.command == 'clean':
        clean_images()
