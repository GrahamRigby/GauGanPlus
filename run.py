from Trainer import run
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-batch_size", type=int, default=24)
parser.add_argument("-img_size", type=int, default=128, help="input/output image height/width, must be square")
parser.add_argument("-training_epochs", type=int, default=200)
parser.add_argument("-save_interval", type=int, default=10, help="saves model sequentially after specified epochs")
parser.add_argument("-train", type=bool, default=True, help="runs model in train mode, setting this to false implies" \
                                                           " the model will be run in eval mode")
parser.add_argument("-load_model_state", type=bool, default=False, help="setting this to true implies the model will" \
                                                                      " start from your last saved chepoint")
args = parser.parse_args()
run(args.batch_size, args.img_size, args.training_epochs, args.save_interval, args.train, args.load_model_state)

