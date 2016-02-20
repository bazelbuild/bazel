package scala.test

object ScalaBinary {
  def main(args: Array[String]) {
    println(MacroTest.hello(1 + 2))
    HelloLib.printMessage("Hello");
  }
}
