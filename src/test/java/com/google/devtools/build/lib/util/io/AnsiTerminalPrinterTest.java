// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.util.io;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.util.io.AnsiTerminalPrinter.Mode;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.google.devtools.build.lib.testutil.MoreAsserts;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayOutputStream;

/**
 * A test for {@link AnsiTerminalPrinter}.
 */
@RunWith(JUnit4.class)
public class AnsiTerminalPrinterTest {
  private ByteArrayOutputStream stream;
  private AnsiTerminalPrinter printer;

  @Before
  public void setUp() throws Exception {
    stream = new ByteArrayOutputStream(1000);
    printer = new AnsiTerminalPrinter(stream, true);
  }

  private void setPlainPrinter() {
    printer = new AnsiTerminalPrinter(stream, false);
  }

  private void assertString(String string) {
    assertEquals(string, stream.toString());
  }

  private void assertRegex(String regex) {
    MoreAsserts.assertStdoutContainsRegex(regex, stream.toString(), "");
  }

  @Test
  public void testPlainPrinter() throws Exception {
    setPlainPrinter();
    printer.print("1" + Mode.INFO + "2" + Mode.ERROR + "3" + Mode.WARNING + "4"
        + Mode.DEFAULT + "5");
    assertString("12345");
  }

  @Test
  public void testDefaultModeIsDefault() throws Exception {
    printer.print("1" + Mode.DEFAULT + "2");
    assertString("12");
  }

  @Test
  public void testDuplicateMode() throws Exception {
    printer.print("_A_" + Mode.INFO);
    printer.print("_B_" + Mode.INFO + "_C_");
    assertRegex("^_A_.+_B__C_$");
  }

  @Test
  public void testModeCodes() throws Exception {
    printer.print(Mode.INFO + "XXX" + Mode.ERROR + "XXX" + Mode.WARNING +"XXX" + Mode.DEFAULT
        + "XXX" + Mode.INFO + "XXX" + Mode.ERROR + "XXX" + Mode.WARNING +"XXX" + Mode.DEFAULT);
    String[] codes = stream.toString().split("XXX");
    assertThat(codes).hasLength(8);
    for (int i = 0; i < 4; i++) {
      assertThat(codes[i]).isNotEmpty();
      assertEquals(codes[i], codes[i+4]);
    }
    assertFalse(codes[0].equals(codes[1]));
    assertFalse(codes[0].equals(codes[2]));
    assertFalse(codes[0].equals(codes[3]));
    assertFalse(codes[1].equals(codes[2]));
    assertFalse(codes[1].equals(codes[3]));
    assertFalse(codes[2].equals(codes[3]));
  }
}
