
# Define the Input Files
InputFiles=["mP1000_mC150_mX100.h5",
            "mP1000_mC400_mX100.h5",
            "mP1000_mC600_mX100.h5",
            "mP1000_mC900_mX100.h5",
            "mP1000_mC200_mX100.h5",
            "mP1000_mC500_mX100.h5",
            "mP1000_mC700_mX100.h5",
            "mP1000_mC950_mX100.h5",
            "mP1000_mC300_mX100.h5",
            "mP1000_mC550_mX100.h5",
            "mP1000_mC800_mX100.h5"]
Files=[]

# Select datasets (formerly TTrees in original ROOT file)


for InputData in InputFiles:
    InputData="/scratch/data-backup/afarbin/crogan/h5/"+InputData
    Files+= [
        [InputData, "AA_Gen"],
        [InputData, "AB_Gen"],
        [InputData, "BA_Gen"],
        [InputData, "BB_Gen"]
    ]

Samples=[]

for F in Files:
    if type(F) != str:
        name=F[1]
        F=F[0]
        Samples.append([F,name])
    else:
        name=F.split(".")[0].split("/")[-1]
        Samples.append([F,name+"_SRAll"])
