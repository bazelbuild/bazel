package com.google.devtools.build.lib.integration.blackbox.framework.junit;

import org.junit.Rule;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TimeoutTestWatcherBaseTest {
  boolean timeoutCaught = false;

  @Rule
  public TimeoutTestWatcher testWatcher =
      new TimeoutTestWatcher() {
        @Override
        protected long getTimeoutMillis() {
          return 100;
        }

        @Override
        protected boolean onTimeout() {
          return timeoutCaught = true;
        }
      };
}
