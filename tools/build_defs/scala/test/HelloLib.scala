package scala.test

object HelloLib {
  def printMessage(arg: String) {
    println(arg + " " + OtherLib.getMessage())
  }
}
