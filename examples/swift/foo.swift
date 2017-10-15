import class examples_BarLib.Multiplier

public class Foo {
  public init() {}

  public func multiply() -> Int {
    return Multiplier().multiply(a: Constants.x, b: Constants.y)
  }
}
