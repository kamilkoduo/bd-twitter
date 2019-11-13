//object DatasetLoader {
//  // Loading the text file using sc.textFile function and creating an RDD
//  // RDD shape: “CleanedText”,Category”
//
//  val input_path = "/path/to/data.txt"
//  val input_RDD = sc.textFile(input_path).map(x => {
//    val row = x.split(",")
//    (cleanDocument(row(1)),row(2))
//  })
//
//  // Converting an RDD to DataFrame
//  val trainingDF = sqlContext.createDataFrame(input_RDD)
//    .toDF("id","cleaned","category")
//
//  // Slicing the data into 70:30 ratio for training and testing data
//  val Array(trainingData, testData) = trainingDF.randomSplit(Array(0.7, 0.3))
//
//  // print the training data
//  trainingData.show()
//}
