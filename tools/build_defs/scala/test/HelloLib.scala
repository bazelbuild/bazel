package scala.test

object HelloLib {
  def printMessage(arg: String) {
    println(arg + " " + OtherLib.getMessage())
    println(arg + " " + OtherJavaLib.getMessage())
  }
}
