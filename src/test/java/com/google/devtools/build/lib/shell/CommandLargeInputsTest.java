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
import static org.junit.Assert.fail;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Tests the command class with large inputs
 *
 */
@RunWith(JUnit4.class)
public class CommandLargeInputsTest {

  @Before
  public void setUp() throws Exception {

    // enable all log statements to ensure there are no problems with
    // logging code
    Logger.getLogger("com.google.devtools.build.lib.shell.Command").setLevel(Level.FINEST);
  }

  private byte[] getRandomBytes() {
    byte[] randomBytes;
    final Random rand = new Random(0xdeadbeef);
    randomBytes = new byte[10000];
    rand.nextBytes(randomBytes);
    return randomBytes;
  }

  private byte[] getAllByteValues() {
    byte[] allByteValues = new byte[Byte.MAX_VALUE - Byte.MIN_VALUE];
    for(int i = 0; i < allByteValues.length; i++) {
      allByteValues[i] = (byte) (i + Byte.MIN_VALUE);
    }
    return allByteValues;
  }

  @Test
  public void testCatRandomBinary() throws Exception {
    final Command command = new Command(new String[] {"cat"});
    byte[] randomBytes = getRandomBytes();
    final CommandResult result = command.execute(randomBytes);
    assertEquals(0, result.getTerminationStatus().getRawResult());
    TestUtil.assertArrayEquals(randomBytes, result.getStdout());
    assertEquals(0, result.getStderr().length);
   }

  @Test
  public void testCatRandomBinaryToOutputStream() throws Exception {
    final Command command = new Command(new String[] {"cat"});
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream err = new ByteArrayOutputStream();
    byte[] randomBytes = getRandomBytes();
    final CommandResult result = command.execute(randomBytes,
                                                 Command.NO_OBSERVER, out, err);
    assertEquals(0, result.getTerminationStatus().getRawResult());
    TestUtil.assertArrayEquals(randomBytes, out.toByteArray());
    assertEquals(0, err.toByteArray().length);
    assertOutAndErrNotAvailable(result);
  }

  @Test
  public void testCatRandomBinaryToErrorStream() throws Exception {
    final Command command = new Command(new String[] {"/bin/sh", "-c", "cat >&2"});
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream err = new ByteArrayOutputStream();
    byte[] randomBytes = getRandomBytes();
    final CommandResult result = command.execute(randomBytes,
                                                 Command.NO_OBSERVER, out, err);
    assertEquals(0, result.getTerminationStatus().getRawResult());
    assertEquals(0, out.toByteArray().length);
    TestUtil.assertArrayEquals(randomBytes, err.toByteArray());
    assertOutAndErrNotAvailable(result);
  }

  @Test
  public void testCatRandomBinaryFromInputStreamToErrorStream()
  throws Exception {
    final Command command = new Command(new String[] {"/bin/sh", "-c", "cat >&2"});
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream err = new ByteArrayOutputStream();
    byte[] randomBytes = getRandomBytes();
    ByteArrayInputStream in = new ByteArrayInputStream(randomBytes);

    final CommandResult result = command.execute(in,
                                                 Command.NO_OBSERVER, out, err);
    assertEquals(0, result.getTerminationStatus().getRawResult());
    assertEquals(0, out.toByteArray().length);
    TestUtil.assertArrayEquals(randomBytes, err.toByteArray());
    assertOutAndErrNotAvailable(result);
  }

  @Test
  public void testStdoutInterleavedWithStdErr() throws Exception {
    final Command command = new Command(new String[]{"/bin/bash",
      "-c", "for i in $( seq 0 999); do (echo OUT$i >&1) && (echo ERR$i  >&2); done"
    });
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ByteArrayOutputStream err = new ByteArrayOutputStream();
    command.execute(Command.NO_INPUT, Command.NO_OBSERVER, out, err);
    StringBuilder expectedOut = new StringBuilder();
    StringBuilder expectedErr = new StringBuilder();
    for (int i = 0; i < 1000; i++) {
      expectedOut.append("OUT").append(i).append("\n");
      expectedErr.append("ERR").append(i).append("\n");
    }
    assertEquals(expectedOut.toString(), out.toString("UTF-8"));
    assertEquals(expectedErr.toString(), err.toString("UTF-8"));
  }

  private void assertOutAndErrNotAvailable(final CommandResult result) {
    try {
      result.getStdout();
      fail();
    } catch (IllegalStateException e){}
    try {
      result.getStderr();
      fail();
    } catch (IllegalStateException e){}
  }

  @Test
  public void testCatAllByteValues() throws Exception {
    final Command command = new Command(new String[] {"cat"});
    byte[] allByteValues = getAllByteValues();
    final CommandResult result = command.execute(allByteValues);
    assertEquals(0, result.getTerminationStatus().getRawResult());
    assertEquals(0, result.getStderr().length);
    TestUtil.assertArrayEquals(allByteValues, result.getStdout());
  }

}
