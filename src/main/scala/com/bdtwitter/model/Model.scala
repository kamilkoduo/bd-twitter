package com.bdtwitter.model

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, ProbabilisticClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, LabeledPoint, RegexTokenizer, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

import scala.io.Source
import scala.util.Try

object Model {
  def getIndexer(): StringIndexer = {
    val indexer = new StringIndexer()
      .setInputCol("sentiment")
      .setOutputCol("label")
    indexer
  }
  def getTokenizer():RegexTokenizer={
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\W")
    tokenizer
  }
  def getStopwordRemover()={
    val src = Source.fromFile("data/stopwords.csv")
    val stopwords = src.getLines().toList
    src.close()
    val stopWordsRemover = new StopWordsRemover()
      .setStopWords(stopwords.toArray)
      .setInputCol("tokens")
      .setOutputCol("clean_tokens")
    stopWordsRemover
  }

  def getHashingTF():HashingTF={
    val hashingTF = new HashingTF()
      .setInputCol("tokens").setOutputCol("features")
    hashingTF
  }
  def getClassifierLogReg()={
    val clf = new LogisticRegression().setMaxIter(100).setRegParam(0.001)
    clf
  }
  def getClassifierDesTr()={
    val clf = new DecisionTreeClassifier()
      .setMaxDepth(5)
    clf
  }

  def getPipelineLogReg()={
    val indexer = getIndexer()
    val tokenizer = getTokenizer()
    val stopWordsRemover = getStopwordRemover()
    val hashingTF = getHashingTF()
    val classifier = getClassifierLogReg()
    val pipeline = new Pipeline().setStages(Array(indexer,tokenizer,stopWordsRemover, hashingTF, classifier))
    pipeline
  }
  def getPipelineDesTr()={
    val indexer = getIndexer()
    val tokenizer = getTokenizer()
    val stopWordsRemover = getStopwordRemover()
    val hashingTF = getHashingTF()
    val classifier = getClassifierDesTr()
    val pipeline = new Pipeline().setStages(Array(indexer,tokenizer,stopWordsRemover, hashingTF, classifier))
    pipeline
  }
  def train(pipeline:Pipeline, trainingData:DataFrame)={
    val model = pipeline.fit(trainingData)
    model
  }
  def predict(model:PipelineModel, testDF:DataFrame)={
    // Create the classification pipeline and train the model
    val prediction = model.transform(testDF)
//      .select("id","label","prediction")

    prediction
  }
  def evaluate(testDF:DataFrame) = {
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
    evaluator.evaluate(testDF)
  }
  def eval_metrics(model:PipelineModel, withPredictions:DataFrame) = {
    val predLabels = withPredictions.select("prediction", "label")

    val predLabelsRDD = predLabels.rdd.map(row => (row(0).toString.toDouble, row(1).toString.toDouble))
    predLabelsRDD.collect().foreach(println)

    val metrics = new BinaryClassificationMetrics(predLabelsRDD)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }


    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val f05Score = metrics.fMeasureByThreshold(beta)
    f05Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }


  }
  def save_model(model:PipelineModel, path:String)={
    model.save(path)
  }
  def save_eval(eval:Double, path:String)={
    import java.io._
    val pw = new PrintWriter(new File(path))
    pw.write(eval.toString)
    pw.close()
  }


  def load_model(path:String):PipelineModel= {
    val model = PipelineModel.load(path)
    model
  }

}
