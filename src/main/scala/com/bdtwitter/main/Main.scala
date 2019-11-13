import com.bdtwitter.preprocessor.Preprocessor
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import Preprocessor.{cleanDocument}

object  Main extends App {
  def Run()={
    // Init Spark
    val conf = new SparkConf().setMaster("local[*]").setAppName("DC")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // Loading the text file using sc.textFile function and creating an RDD
    // RDD shape: “CleanedText”,Category”
    val input_path = "dataset/train.csv"
    val input_RDD = sc.read.format("com.databricks.spark.csv")
      .option("delimiter", ",")
      .load(input_path)

    // Converting an RDD to DataFrame
    val trainingDF = sqlContext.createDataFrame(input_RDD)
      .toDF("id","sentiment","text")

    // Slicing the data into 70:30 ratio for training and testing data
    val Array(trainingData, testData) = trainingDF.randomSplit(Array(0.7, 0.3))

    // print the training data
    trainingData.show()



    println("Hello world!")

  }

  override def main(args: Array[String]): Unit = {
    Run()
  }

}
