import logging # for logging loss history and other info
from model.train import train # training sub-routine
from model.test import test
def main():

    logging.basicConfig(filename='training.log', level=logging.INFO)
    train('config.json',2000,2,debug_mode=False)
    # test('model.pth','config.json')
if __name__ == '__main__':
    main()
