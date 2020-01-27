package com.example.myproject;

import org.junit.Test;

/** Contains a single test that just sleeps for 15 seconds and then succeeds. */
public class TestSleep {
  @Test
  public void testSleep() throws Exception {
    // Sleep for 15 seconds.
    Thread.sleep(15 * 1000);
  }
}
