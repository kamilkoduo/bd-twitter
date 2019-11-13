package com.bdtwitter.preprocessor

import scala.io.Source

object Preprocessor {
  // Building a List of Regex for PreProcessing the text
  def getRegexList(): Map[String, String] = {
    var RegexList = Map[String, String]()
    RegexList += ("punctuation" -> "[^a-zA-Z0-9]")
    RegexList += ("digits" -> "\\b\\d+\\b")
    RegexList += ("white_space" -> "\\s+")
    RegexList += ("small_words" -> "\\b[a-zA-Z0-9]{1,2}\\b")
    RegexList += ("urls" -> "(https?\\://)\\S+")
    RegexList
  }

  // Loading a stopwords list
  def getStopwordsList():Map[String, List[String]] = {
    val stopwordsPath = "data/stopwords.txt"
    var Stopwords = Map[String, List[String]]()
    val src = Source.fromFile(stopwordsPath)
    Stopwords += ("english" -> src.getLines().toList)
    src.close()
    Stopwords
  }

  // Utility function to remove particular regex from text
  def removeRegex(txt: String, regexList:Map[String,String], flag: String): String = {
    val regex = regexList.get(flag)
    var cleaned = txt
    regex match {
      case Some(value) =>
        if (value.equals("white_space")) cleaned = txt.replaceAll(value, "")
        else cleaned = txt.replaceAll(value, " ")
      case None => println("No regex flag matched")
    }
    cleaned
  }

  // Particular function to remove custom word list from text

  def removeCustomWords(txt: String, stopwordsList:Map[String,List[String]],flag: String): String = {
    var words = txt.split(" ")
    val stopwords = stopwordsList.get(flag)
    stopwords match {
      case Some(value) => words = words.filter(x => !value.contains(x))
      case None => println("No stopword flag matched")
    }
    words.mkString(" ")
  }

  // Function to perform step by step text preprocessing and cleaning on documents
  def cleanDocument(document_text: String) : String = {
    val regexList = getRegexList()
    val stopwordsList = getStopwordsList()
    var text = document_text.toLowerCase
    text = removeRegex(text,regexList,"urls")
    text = removeRegex(text,regexList,"punctuation")
    text = removeRegex(text,regexList,"digits")
    text = removeRegex(text,regexList,"small_words")
    text = removeRegex(text,regexList,"white_space")
    text = removeCustomWords(text, stopwordsList, "english")
    text
  }
  def Run(inPath:String,outPath:String) = {
    val src = Source.fromFile(inPath)
    val documentString = src.getLines().mkString(" ")
    src.close()
    val cleaned = cleanDocument(documentString)
    // PrintWriter
    import java.io._
    val pw = new PrintWriter(new File(outPath))
    pw.write(cleaned)
    pw.close()
  }
}
