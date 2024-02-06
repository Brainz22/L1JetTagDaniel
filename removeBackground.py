import h5py
import argparse
import numpy as np

def main(args):
    
    """The order of arguments: training file, testing file, testing sample data, training sample data, 
      and lastly, the tag.  """

    trainFileName = args.trainFileName
    testFileName = args.testFileName
    train_sampleData = args.train_sampleData
    test_sampleData = args.test_sampleData

    print("Reading signal from " + trainFileName)
    print("Reading  from " + testFileName)
    print("Reading signal from " + test_sampleData)
    print("Reading  from " + train_sampleData)
    

    TrainF = h5py.File(trainFileName, "r")
    TestF = h5py.File(testFileName, "r")
    test_sampleF = h5py.File(test_sampleData, "r")
    train_sampleF = h5py.File(train_sampleData, "r")

    TrainData = TrainF["Training Data"][:]
    TestData = TestF["Testing Data"][:]
    train_jetData = train_sampleF["Sample Data"][:]
    test_jetData = test_sampleF["Jet Data"][:]

    Dict = { "Signal in Training" : TrainData[:, -1] == 1, "Signal in Testing" : \
        TestData[:, -1] == 1 }


    newTrainData = TrainData[Dict["Signal in Training"]] #all signal jets in training data
    newTestData = TestData[Dict["Signal in Testing"]] #all signal jets in testing data
    newTrain_jetData = train_jetData[Dict["Signal in Training"]]
    newTest_jetData = test_jetData[Dict["Signal in Testing"]]

    with h5py.File("newTestData" + str(args.tag) + ".h5", "w") as hf:
        hf.create_dataset("Testing Data", data=newTestData)
    with h5py.File("newTrainData" + str(args.tag) + ".h5", "w") as hf:
        hf.create_dataset("Training Data", data=newTrainData)
    with h5py.File("newSampleData" + str(args.tag) + ".h5", "w") as hf:
        hf.create_dataset("Sample Data", data=newTrain_jetData)
    with h5py.File("newJetData" + str(args.tag) + ".h5", "w") as hf:
        hf.create_dataset("Jet Data", data=newTest_jetData)

    print("\n Final number of Number of Signal Jets: ", int(newTrainData.shape[0], "\n"))

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Process arguments")
     parser.add_argument("trainFileName", type=str, help="input .h5 training file path")
     parser.add_argument("testFileName", type=str, help="input .h5 testing file path")
     parser.add_argument("test_sampleData", type=str, help="input .h5 jet data file path")
     parser.add_argument("train_sampleData", type=str, help="input .h5 sample data file path")
     parser.add_argument("tag", type=str, help="Sample type. E.g, ST30: STop with 30 GeV Pt cut")
     
     args = parser.parse_args()

     main(args)
 
