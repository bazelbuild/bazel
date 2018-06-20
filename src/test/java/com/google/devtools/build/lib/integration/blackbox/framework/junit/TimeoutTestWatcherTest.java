package com.google.devtools.build.lib.integration.blackbox.framework.junit;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TimeoutTestWatcherTest extends TimeoutTestWatcherBaseTest {
  @After
  public void tearDown() {
    if ("testTimeoutCaught".equals(testWatcher.getName())) {
      Assert.assertTrue(timeoutCaught);
    }
  }

  /**
   * Test that timeout handler is called
   */
  @Test
  public void testTimeoutCaught() throws Exception {
    for (int i = 0; i < 10; i++) {
      Thread.sleep(500);
    }
  }

  /**
   * Test that normal test failures are not blocked
   */
  @Test(expected = AssertionError.class)
  public void testFailure()  {
    Assert.assertTrue(false);
  }
}
