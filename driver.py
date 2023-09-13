from model import main
import sys
import warnings
warnings.filterwarnings("ignore")

def help():
    print("Commands :")
    print("-h | --help           :   Display Help")
    print("--epoch (1000)        :   Define Number of epoches")
    print("--strength (7500) : Define strength of style")
    print("--style               : Path / URL of style image")
    print("--content             : Path / URL of content image")

if "-h" in sys.argv or "--help" in sys.argv:
    help()
else :
    try :
        send = {}
        if "--epoch" in sys.argv:
            send["numepochs"] = int(sys.argv[sys.argv.index("--epoch")+1])
        if "--content" in sys.argv:
            send["img4content"] = str(sys.argv[sys.argv.index("--content")+1])
        if "--style" in sys.argv:
            send["img4style"] = str(sys.argv[sys.argv.index("--style")+1])
        if "--strength" in sys.argv:
            send["styleScaling"] = int(sys.argv[sys.argv.index("--strength")+1])

        main(**send)
    except Exception as e:
        print()
        print(e)
        print()
        print("SOME ERROR OCCURED !!!")
        help()
        
print()