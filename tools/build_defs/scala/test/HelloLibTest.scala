package scala.test

import org.scalatest._

class ScalaSuite extends FlatSpec {
  "HelloLib" should "call scala" in {
    assert(HelloLib.getOtherLibMessage("hello").equals("hello scala!"))
  }
}

class JavaSuite extends FlatSpec {
  "HelloLib" should "call java" in {
    assert(HelloLib.getOtherJavaLibMessage("hello").equals("hello java!"))
  }
}

