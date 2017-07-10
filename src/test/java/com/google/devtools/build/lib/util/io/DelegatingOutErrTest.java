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
import static com.google.devtools.build.lib.util.StringUtilities.joinLines;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link DelegatingOutErr}.
 */
@RunWith(JUnit4.class)
public class DelegatingOutErrTest {

  private DelegatingOutErr delegate;

  @Before
  public final void createDelegate() {
    delegate = new DelegatingOutErr();
  }

  @After
  public final void closeDelegate() throws Exception {
    delegate.close();
  }

  @Test
  public void testNewDelegateIsLikeDevNull() {
    delegate.printOut("Hello, world.\n");
    delegate.printErr("Feel free to ignore me.\n");
  }

  @Test
  public void testSubscribeAndUnsubscribeSink() {
    delegate.printOut("Nobody will listen to this.\n");
    RecordingOutErr sink = new RecordingOutErr();
    delegate.addSink(sink);
    delegate.printOutLn("Hello, sink.");
    delegate.removeSink(sink);
    delegate.printOutLn("... and alone again ...");
    delegate.addSink(sink);
    delegate.printOutLn("How are things?");
    assertThat(sink.outAsLatin1()).isEqualTo("Hello, sink.\nHow are things?\n");
  }

  @Test
  public void testSubscribeMultipleSinks() {
    RecordingOutErr left = new RecordingOutErr();
    RecordingOutErr right = new RecordingOutErr();
    delegate.addSink(left);
    delegate.printOutLn("left only");
    delegate.addSink(right);
    delegate.printOutLn("both");
    delegate.removeSink(left);
    delegate.printOutLn("right only");
    delegate.removeSink(right);
    delegate.printOutLn("silence");
    delegate.addSink(left);
    delegate.addSink(right);
    delegate.printOutLn("left and right");
    assertThat(left.outAsLatin1()).isEqualTo(joinLines("left only", "both", "left and right", ""));
    assertThat(right.outAsLatin1())
        .isEqualTo(joinLines("both", "right only", "left and right", ""));
  }
}
