// Copyright 2014 The Bazel Authors. All rights reserved.
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

import java.io.ByteArrayOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link OutErr}. */
@RunWith(JUnit4.class)
public class OutErrTest {

  private ByteArrayOutputStream out = new ByteArrayOutputStream();
  private ByteArrayOutputStream err = new ByteArrayOutputStream();
  private OutErr outErr = OutErr.create(out, err);

  @Test
  public void testRetainsOutErr() {
    assertThat(outErr.getOutputStream()).isSameInstanceAs(out);
    assertThat(outErr.getErrorStream()).isSameInstanceAs(err);
  }

  @Test
  public void testPrintsToOut() {
    outErr.printOut("Hello, world.");
    assertThat(new String(out.toByteArray())).isEqualTo("Hello, world.");
  }

  @Test
  public void testPrintsToErr() {
    outErr.printErr("Hello, moon.");
    assertThat(new String(err.toByteArray())).isEqualTo("Hello, moon.");
  }

  @Test
  public void testPrintsToOutWithANewline() {
    outErr.printOutLn("With a newline.");
    assertThat(new String(out.toByteArray())).isEqualTo("With a newline.\n");
  }

  @Test
  public void testPrintsToErrWithANewline() {
    outErr.printErrLn("With a newline.");
    assertThat(new String(err.toByteArray())).isEqualTo("With a newline.\n");
  }

  @Test
  public void testPrintsTwoLinesToOut() {
    outErr.printOutLn("line 1");
    outErr.printOutLn("line 2");
    assertThat(new String(out.toByteArray())).isEqualTo("line 1\nline 2\n");
  }

  @Test
  public void testPrintsTwoLinesToErr() {
    outErr.printErrLn("line 1");
    outErr.printErrLn("line 2");
    assertThat(new String(err.toByteArray())).isEqualTo("line 1\nline 2\n");
  }

}
