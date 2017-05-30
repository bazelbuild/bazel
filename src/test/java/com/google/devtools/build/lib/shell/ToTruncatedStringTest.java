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

import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link LogUtil#toTruncatedString}. */
/*
 * Note: The toTruncatedString method uses the platform encoding intentionally,
 * so the unittest does to. Check out the comment in the implementation in
 * case you're wondering why.
 */
@RunWith(JUnit4.class)
public class ToTruncatedStringTest {

  @Before
  public final void configureLogger() throws Exception  {
    // enable all log statements to ensure there are no problems with
    // logging code
    Logger.getLogger("com.google.devtools.build.lib.shell.Command").setLevel(Level.FINEST);
  }

  @Test
  public void testTruncatingNullYieldsEmptyString() {
    assertThat(LogUtil.toTruncatedString(null)).isEmpty();
  }

  @Test
  public void testTruncatingEmptyArrayYieldsEmptyString() {
    assertThat(LogUtil.toTruncatedString(new byte[0])).isEmpty();
  }

  @Test
  public void testTruncatingSampleArrayYieldsTruncatedString() {
    String sampleInput = "Well, there could be a lot of output, but we want " +
            "to produce a useful log. A log is useful if it contains the " +
            "interesting information (like what the command was), and maybe " +
            "some of the output. However, too much is too much, so we just " +
            "cut it after 150 bytes ...";
    String expectedOutput = "Well, there could be a lot of output, but we " +
            "want to produce a useful log. A log is useful if it contains " +
            "the interesting information (like what the c[... truncated. " +
            "original size was 261 bytes.]";
    assertThat(LogUtil.toTruncatedString(sampleInput.getBytes())).isEqualTo(expectedOutput);
  }

  @Test
  public void testTruncatingHelloWorldYieldsHelloWorld() {
    String helloWorld = "Hello, world.";
    assertThat(LogUtil.toTruncatedString(helloWorld.getBytes())).isEqualTo(helloWorld);
  }

}
