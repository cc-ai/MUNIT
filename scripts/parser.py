>>>>>
<<<<<<
# Parse the different arguments
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", type=int, default=0, help="cuda device")
>>>>>
<<<<<<
opts = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=opts.gpu