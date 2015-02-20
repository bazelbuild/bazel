package com.example.myproject;

import org.junit.Assert;
import org.junit.Test;

/**
 * A test that always fails.
 */
public class Fail {
  @Test
  public void testFail() {
    Assert.fail("This is an expected test failure.");
  }
}
