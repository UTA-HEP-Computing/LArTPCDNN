# Configuration of this job
parser = argparse.ArgumentParser()
parser.add_argument('-C', '--config',default="EventClassificationDNN/ScanConfig.py")
parser.add_argument('-I', '--inputdata',default="/data/afarbin/crogan/h5/mP1000_mC200_mX100.h5")
parser.add_argument('-L', '--LoadModel',default=False)
parser.add_argument('--gpu', dest='gpuid', default="")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--NoTrain', action="store_true")
parser.add_argument('-s',"--hyperparamset", default="0")
parser.add_argument('-v',"--varset", default="0")
parser.add_argument('--NoResults', action="store_false")

args = parser.parse_args()

UseGPU=not args.cpu
gpuid=args.gpuid
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)

if args.varset:
    VarSet = int(args.varset)

    print "Using VarSet: ",VarSet

ConfigFile=args.config
InputData=args.inputdata

LoadModel=args.LoadModel

# Configuration from PBS:
if "PBS_ARRAYID" in os.environ:
    HyperParamSet = int(os.environ["PBS_ARRAYID"])

if "PBS_QUEUE" in os.environ:
    if "cpu" in os.environ["PBS_QUEUE"]:
        UseGPU=False
    if "gpu" in os.environ["PBS_QUEUE"]:
        UseGPU=True
        gpuid=int(os.environ["PBS_QUEUE"][3:4])

if UseGPU:
    print "Using GPU",gpuid
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu%s,floatX=float32,force_device=True" % (gpuid)
else:
    print "Using CPU."

Train= not args.NoTrain

WriteResults=  args.NoResults
