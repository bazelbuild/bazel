// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.shell;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests of the interaction of Thread.interrupt and Command.execute.
 *
 * Read http://www-128.ibm.com/developerworks/java/library/j-jtp05236.html
 * for background material.
 *
 * NOTE: This test is dependent on thread timings.  Under extreme machine load
 * it's possible that this test could fail spuriously or intermittently.  In
 * that case, adjust the timing constants to increase the tolerance.
 */
@RunWith(JUnit4.class)
public class InterruptibleTest {

  private final Thread mainThread = Thread.currentThread();

  // Interrupt main thread after 1 second.  Hopefully by then /bin/sleep
  // should be running.
  private final Thread interrupter = new Thread() {
      @Override
      public void run() {
        try {
          Thread.sleep(1000); // 1 sec
        } catch (InterruptedException e) {
          throw new IllegalStateException("Unexpected interrupt!");
        }
        mainThread.interrupt();
      }
    };

  private Command command;
  private long before;

  @Before
  public void setUp() throws Exception {

    Thread.interrupted(); // side effect: clear interrupted status
    assertFalse("Unexpected interruption!", mainThread.isInterrupted());

    this.command = new Command(new String[] { "/bin/sleep", "2" });
    this.before = System.nanoTime();

    interrupter.start();
  }

  @After
  public void tearDown() throws Exception {
    interrupter.join();
    Thread.interrupted(); // Clear interrupted status, or else other tests may fail.
  }

  private void assertDuration(long minMillis, long maxMillis)
      throws Exception {
    long after = System.nanoTime();
    long millis = (after - before) / 1000000;
    if (millis < minMillis || millis > maxMillis) {
      fail("Duration " + millis + "ms was not in range "
           + minMillis + "-" + maxMillis + "ms");
    }
  }

  /**
   * Test that interrupting a thread in an "uninterruptible" Command.execute
   * preserves the thread's interruptible status, and does not terminate the
   * subprocess.
   */
  @Test
  public void testUninterruptibleCommandRunsToCompletion() throws Exception {
    command.execute();

    // Subprocess execution should be around 2000ms:
    assertDuration(2000, 2500);

    // The interrupter thread should have exited about 1000ms ago.
    assertFalse("Interrupter thread is still alive!",
                interrupter.isAlive());

    // The interrupter thread should have set the main thread's interrupt flag.
    assertTrue("Main thread was not interrupted during command execution!",
               mainThread.isInterrupted());
  }

  /**
   * Test that interrupting a thread in an "interruptible" Command.execute
   * causes preserves the thread's interruptible status, terminates the
   * subprocess, and returns promptly.
   */
  @Test
  public void testInterruptibleCommand() throws Exception {
    try {
      command.execute(Command.NO_INPUT,
                      Command.NO_OBSERVER,
                      System.out,
                      System.err,
                      true); // => interruptible
      fail("Subprocess not aborted!");
    } catch (AbnormalTerminationException e) {
      assertEquals("Process terminated by signal 15", // SIGINT
                   e.getMessage());
    }

    // Subprocess execution should be around 1000ms, but less than 2000, the subprocess sleep time.
    assertDuration(1000, 1900);

    // We don't assert that the interrupter thread has exited; due to prompt
    // termination it might still be running.

    // The interrupter thread should have set the main thread's interrupt flag.
    assertTrue("Main thread was not interrupted during command execution!",
               mainThread.isInterrupted());

  }
}
