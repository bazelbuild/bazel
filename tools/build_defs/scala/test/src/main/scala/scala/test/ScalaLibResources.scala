package scala.test

object ScalaLibResources {
  def getGreetings() = scala.io.Source.fromInputStream(getClass.getResourceAsStream("hellos")).getLines.toList
}
