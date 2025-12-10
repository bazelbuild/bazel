package com.example.myproject;

import org.junit.Test;

/** Contains a single test that just sleeps for 5 seconds and then succeeds. */
public class TestSleep {
  @Test
  public void testSleep() throws Exception {
    // Sleep for 5 seconds.
    Thread.sleep(5 * 1000);
  }
}
