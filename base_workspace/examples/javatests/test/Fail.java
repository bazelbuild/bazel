package test;

import org.junit.Assert;
import org.junit.Test;

public class Fail {
  @Test
  public void testFail() {
    Assert.fail("This is an expected test failure.");
  }
}
