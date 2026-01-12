from spinup import td3
from env import RoboboIREnv


def main():
    env_fn = lambda: RoboboIREnv()
    td3(env_fn, epochs=50)


if __name__ == '__main__':
    main()
