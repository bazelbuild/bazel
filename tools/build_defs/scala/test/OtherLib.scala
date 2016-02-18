package scala.test

// It is just to show how a Scala library can depend on another Scala library.
object OtherLib {
  def getMessage(): String = {
    // This won't compile because Exported is exported, not a dep:
    // Exported.message
    "scala!"
  }
}
