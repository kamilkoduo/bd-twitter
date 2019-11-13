package com.bdtwitter.model

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, ProbabilisticClassifier}
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.sql.{DataFrame, Row}

object Model {
  def getIndexer(): StringIndexer = {
    val indexer = new StringIndexer()
      .setInputCol("sentiment")
      .setOutputCol("label")
    indexer
  }
  def getTokenizer():Tokenizer={
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
    tokenizer
  }
  def getHashingTF():HashingTF={
    val hashingTF = new HashingTF()
      .setInputCol("tokens").setOutputCol("features")
    hashingTF
  }
  def getClassifier()={
    val clf = new LogisticRegression().setMaxIter(100).setRegParam(0.001)
    clf
  }
  def getPipeline()={
    val indexer = getIndexer()
    val tokenizer = getTokenizer()
    val hashingTF = getHashingTF()
    val classifier = getClassifier()
    val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, hashingTF, classifier))
    pipeline
  }
  def train(trainingData:DataFrame)={
    val pipeline = getPipeline()
    val model = pipeline.fit(trainingData)
    model
  }
  def predict(model:)={
    // Create the classification pipeline and train the model
    val prediction = model.transform(testData).select("id","cleaned_text","category","prediction")

    // Print the predictions
    prediction.foreach(println)
  }
}
