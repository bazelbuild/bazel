// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.shell.TestUtil.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collections;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Unit tests for {@link Command}. This test will only succeed on Linux,
 * currently, because of its non-portable nature.
 */
@RunWith(JUnit4.class)
public class CommandTest {

  private static final long LONG_TIME = 10000;
  private static final long SHORT_TIME = 250;

  // Platform-independent tests ----------------------------------------------

  @Before
  public final void configureLogger() throws Exception  {
    // enable all log statements to ensure there are no problems with
    // logging code
    Logger.getLogger("com.google.devtools.build.lib.shell.Command").setLevel(Level.FINEST);
  }

  @Test
  public void testIllegalArgs() throws Exception {

    try {
      new Command((String[]) null);
      fail("Should have thrown IllegalArgumentException");
    } catch (IllegalArgumentException iae) {
      // good
    }

    try {
      new Command(new String[] {"/bin/true", null}).execute();
      fail("Should have thrown NullPointerException");
    } catch (NullPointerException npe) {
      // good
    }

    try {
      new Command(new String[] {"foo"}).execute(null);
      fail("Should have thrown NullPointerException");
    } catch (NullPointerException npe) {
      // good
    }

  }

  @Test
  public void testProcessBuilderConstructor() throws Exception {
    String helloWorld = "Hello, world";
    ProcessBuilder builder = new ProcessBuilder("/bin/echo", helloWorld);
    byte[] stdout = new Command(builder).execute().getStdout();
    assertEquals(helloWorld + '\n', new String(stdout, "UTF-8"));
  }

  @Test
  public void testGetters() {
    final File workingDir = new File(".");
    final Map<String,String> env = Collections.singletonMap("foo", "bar");
    final String[] commandArgs = new String[] { "command" };
    final Command command = new Command(commandArgs, env, workingDir);
    assertArrayEquals(commandArgs, command.getCommandLineElements());
    for (final String key : env.keySet()) {
      assertThat(command.getEnvironmentVariables()).containsEntry(key, env.get(key));
    }
    assertEquals(workingDir, command.getWorkingDirectory());
  }

  // Platform-dependent tests ------------------------------------------------

  @Test
  public void testSimpleCommand() throws Exception {
    final Command command = new Command(new String[] {"ls"});
    final CommandResult result = command.execute();
    assertTrue(result.getTerminationStatus().success());
    assertEquals(0, result.getStderr().length);
    assertThat(result.getStdout().length).isGreaterThan(0);
  }

  @Test
  public void testArguments() throws Exception {
    final Command command = new Command(new String[] {"echo", "foo"});
    checkSuccess(command.execute(), "foo\n");
  }

  @Test
  public void testEnvironment() throws Exception {
    final Map<String,String> env = Collections.singletonMap("FOO", "BAR");
    final Command command = new Command(new String[] {"/bin/sh", "-c", "echo $FOO"}, env,
        null);
    checkSuccess(command.execute(), "BAR\n");
  }

  @Test
  public void testWorkingDir() throws Exception {
    final Command command = new Command(new String[] {"pwd"}, null, new File("/"));
    checkSuccess(command.execute(), "/\n");
  }

  @Test
  public void testStdin() throws Exception {
    final Command command = new Command(new String[] {"grep", "bar"});
    checkSuccess(command.execute("foobarbaz".getBytes()), "foobarbaz\n");
  }

  @Test
  public void testRawCommand() throws Exception {
    final Command command = new Command(new String[] { "perl",
                                                       "-e",
                                                       "print 'a'x100000" });
    final CommandResult result = command.execute();
    assertTrue(result.getTerminationStatus().success());
    assertEquals(0, result.getStderr().length);
    assertThat(result.getStdout().length).isGreaterThan(0);
  }

  @Test
  public void testRawCommandWithDir() throws Exception {
    final Command command = new Command(new String[] { "pwd" },
                                        null,
                                        new File("/"));
    final CommandResult result = command.execute();
    checkSuccess(result, "/\n");
  }

  @Test
  public void testHugeOutput() throws Exception {
    final Command command = new Command(new String[] {"perl", "-e", "print 'a'x100000"});
    final CommandResult result = command.execute();
    assertTrue(result.getTerminationStatus().success());
    assertEquals(0, result.getStderr().length);
    assertEquals(100000, result.getStdout().length);
  }

  @Test
  public void testIgnoreOutput() throws Exception {
    final Command command = new Command(new String[] {"perl", "-e", "print 'a'x100000"});
    final CommandResult result = command.execute(Command.NO_INPUT, null, true);
    assertTrue(result.getTerminationStatus().success());
    try {
      result.getStdout();
      fail("Should have thrown IllegalStateException");
    } catch (IllegalStateException ise) {
      // good
    }
    try {
      result.getStderr();
      fail("Should have thrown IllegalStateException");
    } catch (IllegalStateException ise) {
      // good
    }
  }

  @Test
  public void testNoStreamingInputForCat() throws Exception {
    final Command command = new Command(new String[]{"/bin/cat"});
    ByteArrayInputStream emptyInput = new ByteArrayInputStream(new byte[0]);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream err = new ByteArrayOutputStream();
    CommandResult result = command.execute(emptyInput,
                                           Command.NO_OBSERVER, out, err);
    assertTrue(result.getTerminationStatus().success());
    assertThat(out.toString("UTF-8")).isEmpty();
    assertThat(err.toString("UTF-8")).isEmpty();
  }

  @Test
  public void testNoInputForCat() throws Exception {
    final Command command = new Command(new String[]{"/bin/cat"});
    CommandResult result = command.execute();
    assertTrue(result.getTerminationStatus().success());
    assertThat(new String(result.getStdout(), "UTF-8")).isEmpty();
    assertThat(new String(result.getStderr(), "UTF-8")).isEmpty();
  }

  @Test
  public void testProvidedOutputStreamCapturesHelloWorld() throws Exception {
    String helloWorld = "Hello, world.";
    final Command command = new Command(new String[]{"/bin/echo", helloWorld});
    ByteArrayOutputStream stdOut = new ByteArrayOutputStream();
    ByteArrayOutputStream stdErr = new ByteArrayOutputStream();
    command.execute(Command.NO_INPUT, Command.NO_OBSERVER, stdOut, stdErr);
    assertEquals(helloWorld + "\n", stdOut.toString("UTF-8"));
    assertEquals(0, stdErr.toByteArray().length);
  }

  @Test
  public void testAsynchronous() throws Exception {
    final File tempFile = File.createTempFile("googlecron-test", "tmp");
    tempFile.delete();
    final Command command = new Command(new String[] {"touch", tempFile.getAbsolutePath()});
    // Shouldn't throw any exceptions:
    FutureCommandResult result =
        command.executeAsynchronously(Command.NO_INPUT);
    result.get();
    assertTrue(tempFile.exists());
    assertTrue(result.isDone());
    tempFile.delete();
  }

  @Test
  public void testAsynchronousWithKillable() throws Exception {
    final Command command = new Command(new String[] {"sleep", "5"});
    final SimpleKillableObserver observer = new SimpleKillableObserver();
    FutureCommandResult result =
        command.executeAsynchronously(Command.NO_INPUT, observer);
    assertFalse(result.isDone());
    observer.kill();
    try {
      result.get();
    } catch (AbnormalTerminationException e) {
      // Expects, but does not insist on termination with a signal.
    }
    assertTrue(result.isDone());
  }

  @Test
  public void testAsynchronousWithOutputStreams() throws Exception {

    final String helloWorld = "Hello, world.";
    final Command command = new Command(new String[]{"/bin/echo", helloWorld});
    final ByteArrayInputStream emptyInput =
      new ByteArrayInputStream(new byte[0]);
    final ByteArrayOutputStream stdOut = new ByteArrayOutputStream();
    final ByteArrayOutputStream stdErr = new ByteArrayOutputStream();
    FutureCommandResult result = command.executeAsynchronously(emptyInput,
        Command.NO_OBSERVER,
        stdOut,
        stdErr);
    result.get(); // Make sure the process actually finished
    assertEquals(helloWorld + "\n", stdOut.toString("UTF-8"));
    assertEquals(0, stdErr.toByteArray().length);
  }

  @Test
  public void testSimpleKillableObserver() throws Exception {
    final Command command = new Command(new String[] {"sleep", "5"});
    final SimpleKillableObserver observer = new SimpleKillableObserver();
    new Thread() {
      @Override
      public void run() {
        try {
          command.execute(Command.NO_INPUT, observer, true);
          fail();
        } catch (CommandException e) {
          // Good.
          checkCommandElements(e, "sleep", "5");
        }
      }
    }.start();
    // We're racing against the actual startup of the other command. Wait for 10ms so it can start.
    Thread.sleep(10);
    observer.kill();
  }

  @Test
  public void testTimeout() throws Exception {
    // Sleep for 3 seconds,
    final Command command = new Command(new String[] {"sleep", "3"});
    try {
      // but timeout after 1 second
      command.execute(Command.NO_INPUT, 1000L, false);
      fail("Should have thrown AbnormalTerminationException");
    } catch (AbnormalTerminationException ate) {
      // good
      checkCommandElements(ate, "sleep", "3");
      checkATE(ate);
    }
  }

  @Test
  public void testTimeoutDoesntFire() throws Exception {
    final Command command = new Command(new String[] {"cat"});
    command.execute(new byte[]{'H', 'i', '!'}, 2000L, false);
  }

  @Test
  public void testCommandDoesNotExist() throws Exception {
    final Command command = new Command(new String[]{"thisisnotreal"});
    try {
      command.execute();
      fail();
    } catch (ExecFailedException e){
      // Good.
      checkCommandElements(e, "thisisnotreal");
    }
  }

  @Test
  public void testNoSuchCommand() throws Exception {
    final Command command = new Command(new String[] {"thisisnotreal"});
    try {
      command.execute();
      fail("Should have thrown ExecFailedException");
    } catch (ExecFailedException expected) {
      // good
    }
  }

  @Test
  public void testExitCodes() throws Exception {
    // 0 => success
    {
      String args[] = { "/bin/sh", "-c", "exit 0" };
      CommandResult result = new Command(args).execute();
      TerminationStatus status = result.getTerminationStatus();
      assertTrue(status.success());
      assertTrue(status.exited());
      assertEquals(0, status.getExitCode());
    }

    // Every exit value in range [1-255] is reported as such (except [129-191],
    // which map to signals).
    for (int exit : new int[] { 1, 2, 3, 127, 128, 192, 255 }) {
      try {
        String args[] = { "/bin/sh", "-c", "exit " + exit };
        new Command(args).execute();
        fail("Should have exited with status " + exit);
      } catch (BadExitStatusException e) {
        assertThat(e).hasMessage("Process exited with status " + exit);
        checkCommandElements(e, "/bin/sh", "-c", "exit " + exit);
        TerminationStatus status = e.getResult().getTerminationStatus();
        assertFalse(status.success());
        assertTrue(status.exited());
        assertEquals(exit, status.getExitCode());
        assertEquals("Exit " + exit , status.toShortString());
      }
    }

    // negative exit values are modulo 256:
    for (int exit : new int[] { -1, -2, -3 }) {
      int expected = 256 + exit;
      try {
        String args[] = { "/bin/bash", "-c", "exit " + exit };
        new Command(args).execute();
        fail("Should have exited with status " + expected);
      } catch (BadExitStatusException e) {
        assertThat(e).hasMessage("Process exited with status " + expected);
        checkCommandElements(e, "/bin/bash", "-c", "exit " + exit);
        TerminationStatus status = e.getResult().getTerminationStatus();
        assertFalse(status.success());
        assertTrue(status.exited());
        assertEquals(expected, status.getExitCode());
        assertEquals("Exit " + expected, status.toShortString());
      }
    }
  }

  @Test
  public void testFailedWithSignal() throws Exception {
    // SIGHUP, SIGINT, SIGKILL, SIGTERM
    for (int signal : new int[] { 1, 2, 9, 15 }) {
      // Invoke a C++ program (killmyself.cc) that will die
      // with the specified signal.
      String killmyself = BlazeTestUtils.runfilesDir() + "/"
          + TestConstants.JAVATESTS_ROOT
          + "/com/google/devtools/build/lib/shell/killmyself";
      try {
        String args[] = { killmyself, "" + signal };
        new Command(args).execute();
        fail("Expected signal " + signal);
      } catch (AbnormalTerminationException e) {
        assertThat(e).hasMessage("Process terminated by signal " + signal);
        checkCommandElements(e, killmyself, "" + signal);
        TerminationStatus status = e.getResult().getTerminationStatus();
        assertFalse(status.success());
        assertFalse(status.exited());
        assertEquals(signal, status.getTerminatingSignal());

        switch (signal) {
          case 1:  assertEquals("Hangup",     status.toShortString()); break;
          case 2:  assertEquals("Interrupt",  status.toShortString()); break;
          case 9:  assertEquals("Killed",     status.toShortString()); break;
          case 15: assertEquals("Terminated", status.toShortString()); break;
        }
      }
    }
  }

  @Test
  public void testDestroy() throws Exception {

    // Sleep for 10 seconds,
    final Command command = new Command(new String[] {"sleep", "10"});

    // but kill it after 1
    final KillableObserver killer = new KillableObserver() {
      @Override
      public void startObserving(final Killable killable) {
        final Thread t = new Thread() {
          @Override
          public void run() {
            try {
              Thread.sleep(1000L);
            } catch (InterruptedException ie) {
              // continue
            }
            killable.kill();
          }
        };
        t.start();
      }
      @Override
      public void stopObserving(final Killable killable) {
        // do nothing
      }
    };

    try {
      command.execute(Command.NO_INPUT, killer, false);
      fail("Should have thrown AbnormalTerminationException");
    } catch (AbnormalTerminationException ate) {
      // Good.
      checkCommandElements(ate, "sleep", "10");
      checkATE(ate);
    }
  }

  @Test
  public void testOnlyReadsPartialInput() throws Exception {
    // -c == --bytes, but -c also works on Darwin.
    Command command = new Command(new String[] {"head", "-c", "500"});
    OutputStream out = new ByteArrayOutputStream();
    InputStream in = new InputStream() {

      @Override
      public int read() {
        return 0; // write an unbounded amount
      }

    };

    CommandResult result = command.execute(in, Command.NO_OBSERVER, out, out);
    TerminationStatus status = result.getTerminationStatus();
    assertTrue(status.success());
  }

  @Test
  public void testFlushing() throws Exception {
    final Command command = new Command(
        // On darwin, /bin/sh does not support -n for the echo builtin.
        new String[] {"/bin/bash", "-c", "echo -n Foo; sleep 0.1; echo Bar"});
    // We run this command, passing in a special output stream
    // that records when each flush() occurs.
    // We test that a flush occurs after writing "Foo"
    // and that another flush occurs after writing "Bar\n".
    final boolean[] flushed = new boolean[8];
    OutputStream out = new OutputStream() {
      private int count = 0;
      @Override
      public void write(int b) throws IOException {
        count++;
      }
      @Override
      public void flush() throws IOException {
        flushed[count] = true;
      }
    };
    command.execute(Command.NO_INPUT, Command.NO_OBSERVER, out, System.err);
    assertFalse(flushed[0]);
    assertFalse(flushed[1]); // 'F'
    assertFalse(flushed[2]); // 'o'
    assertTrue(flushed[3]);  // 'o'   <- expect flush here.
    assertFalse(flushed[4]); // 'B'
    assertFalse(flushed[5]); // 'a'
    assertFalse(flushed[6]); // 'r'
    assertTrue(flushed[7]);  // '\n'
  }

  // See also InterruptibleTest.
  @Test
  public void testInterrupt() throws Exception {

    // Sleep for 10 seconds,
    final Command command = new Command(new String[] {"sleep", "10"});
    // Easy but hacky way to let this thread "return" a result to this method
    final CommandResult[] resultContainer = new CommandResult[1];
    final Thread commandThread = new Thread() {
      @Override
      public void run() {
        try {
          resultContainer[0] = command.execute();
        } catch (CommandException ce) {
          fail(ce.toString());
        }
      }
    };
    commandThread.start();

    Thread.sleep(1000L);

    // but interrupt it after 1
    commandThread.interrupt();

    // should continue to wait and exit normally
    commandThread.join();

    final CommandResult result = resultContainer[0];
    assertTrue(result.getTerminationStatus().success());
    assertEquals(0, result.getStderr().length);
    assertEquals(0, result.getStdout().length);
  }

  @Test
  public void testOutputStreamThrowsException() throws Exception {
    OutputStream out = new OutputStream () {
      @Override
      public void write(int b) throws IOException {
        throw new IOException();
      }
    };
    Command command = new Command(new String[] {"/bin/echo", "foo"});
    try {
      command.execute(Command.NO_INPUT, Command.NO_OBSERVER, out, out);
      fail();
    } catch (AbnormalTerminationException e) {
      // Good.
      checkCommandElements(e, "/bin/echo", "foo");
      assertThat(e).hasMessage("java.io.IOException");
    }
  }

  @Test
  public void testOutputStreamThrowsExceptionAndCommandFails()
  throws Exception {
    OutputStream out = new OutputStream () {
      @Override
      public void write(int b) throws IOException {
        throw new IOException();
      }
    };
    Command command = new Command(new String[] {"cat", "/dev/thisisnotreal"});
    try {
      command.execute(Command.NO_INPUT, Command.NO_OBSERVER, out, out);
      fail();
    } catch (AbnormalTerminationException e) {
      checkCommandElements(e, "cat", "/dev/thisisnotreal");
      TerminationStatus status = e.getResult().getTerminationStatus();
      // Subprocess either gets a SIGPIPE trying to write to our output stream,
      // or it exits with failure.  Both are observed, nondetermistically.
      assertTrue(status.exited()
                 ? status.getExitCode() == 1
                 : status.getTerminatingSignal() == 13);
      assertTrue(e.getMessage(),
          e.getMessage().endsWith("also encountered an error while attempting "
                                  + "to retrieve output"));
    }
  }

  /**
   * Helper to test KillableObserver classes.
   */
  private class KillableTester implements Killable {
    private boolean isKilled = false;
    private boolean timedOut = false;
    @Override
    public synchronized void kill() {
      isKilled = true;
      notifyAll();
    }
    public synchronized boolean getIsKilled() {
      return isKilled;
    }
    /**
     * Wait for a specified time or until the {@link #kill()} is called.
     */
    public synchronized void sleepUntilKilled(final long timeoutMS) {
      long nowTime = System.currentTimeMillis();
      long endTime = nowTime + timeoutMS;
      while (!isKilled && !timedOut) {
        long waitTime = endTime - nowTime;
        if (waitTime <= 0) {
          // Process has timed out, needs killing.
          timedOut = true;
          break;
        }
        try {
          wait(waitTime); // Suffers "spurious wakeup", hence the while() loop.
          nowTime = System.currentTimeMillis();
        } catch (InterruptedException exception) {
          break;
        }
      }
    }
  }

  @Test
  public void testTimeOutKillableObserverNoKill() throws Exception {
    KillableTester killable = new KillableTester();
    TimeoutKillableObserver observer = new TimeoutKillableObserver(LONG_TIME);
    observer.startObserving(killable);
    observer.stopObserving(killable);
    assertFalse(observer.hasTimedOut());
    assertFalse(killable.getIsKilled());
  }

  @Test
  public void testTimeOutKillableObserverNoKillWithDelay() throws Exception {
    KillableTester killable = new KillableTester();
    TimeoutKillableObserver observer = new TimeoutKillableObserver(LONG_TIME);
    observer.startObserving(killable);
    killable.sleepUntilKilled(SHORT_TIME);
    observer.stopObserving(killable);
    assertFalse(observer.hasTimedOut());
    assertFalse(killable.getIsKilled());
  }

  @Test
  public void testTimeOutKillableObserverWithKill() throws Exception {
    KillableTester killable = new KillableTester();
    TimeoutKillableObserver observer = new TimeoutKillableObserver(SHORT_TIME);
    observer.startObserving(killable);
    killable.sleepUntilKilled(LONG_TIME);
    observer.stopObserving(killable);
    assertTrue(observer.hasTimedOut());
    assertTrue(killable.getIsKilled());
  }

  @Test
  public void testTimeOutKillableObserverWithKillZeroMillis() throws Exception {
    KillableTester killable = new KillableTester();
    TimeoutKillableObserver observer = new TimeoutKillableObserver(0);
    observer.startObserving(killable);
    killable.sleepUntilKilled(LONG_TIME);
    observer.stopObserving(killable);
    assertTrue(observer.hasTimedOut());
    assertTrue(killable.getIsKilled());
  }

  private static void checkCommandElements(CommandException e,
      String... expected) {
    assertArrayEquals(expected, e.getCommand().getCommandLineElements());
  }

  private static void checkATE(final AbnormalTerminationException ate) {
    final CommandResult result = ate.getResult();
    assertFalse(result.getTerminationStatus().success());
  }

  private static void checkSuccess(final CommandResult result,
                                   final String expectedOutput) {
    assertTrue(result.getTerminationStatus().success());
    assertEquals(0, result.getStderr().length);
    assertEquals(expectedOutput, new String(result.getStdout()));
  }

  @Test
  public void testRelativePath() throws Exception {
    Command command = new Command(new String[]{"relative/path/to/binary"},
        ImmutableMap.<String, String>of(),
        new File("/working/directory"));
    assertThat(command.getCommandLineElements()[0])
        .isEqualTo("/working/directory/relative/path/to/binary");
  }
}
