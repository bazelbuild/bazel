package scala.test

object HelloLib {
  def printMessage(arg: String) {
    println(getOtherLibMessage(arg))
    println(getOtherJavaLibMessage(arg))
  }

  def getOtherLibMessage(arg: String) : String = {
    arg + " " + OtherLib.getMessage()
  }

  def getOtherJavaLibMessage(arg: String) : String = {
    arg + " " + OtherJavaLib.getMessage()
  }
}
