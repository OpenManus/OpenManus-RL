from action1 import AI
import sys

if __name__ == "__main__":
    stage = int(sys.argv[1])
    
    myAI = AI(stage)
    myAI.run()
