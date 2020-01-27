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
package com.google.devtools.build.lib.events;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.io.PrintWriter;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ReporterStreamTest {

  private Reporter reporter;
  private StringBuilder out;
  private EventHandler outAppender;

  @Before
  public final void createOutputAppender() throws Exception  {
    reporter = new Reporter(new EventBus());
    out = new StringBuilder();
    outAppender = new EventHandler() {
      @Override
      public void handle(Event event) {
        out.append("[" + event.getKind() + ": " + event.getMessage() + "]\n");
      }
    };
  }

  @Test
  public void reporterStream() throws Exception {
    assertThat(out.toString()).isEmpty();
    reporter.addHandler(outAppender);
    try (
      PrintWriter warnWriter =
        new PrintWriter(new ReporterStream(reporter, EventKind.WARNING), true);
      PrintWriter infoWriter =
          new PrintWriter(new ReporterStream(reporter, EventKind.INFO), true)
    ) {
      infoWriter.println("some info");
      warnWriter.println("a warning");
    }
    reporter.getOutErr().printOutLn("some output");
    reporter.getOutErr().printErrLn("an error");
    MoreAsserts.assertEqualsUnifyingLineEnds(
        "[INFO: some info\n]\n"
            + "[WARNING: a warning\n]\n"   
            + "[STDOUT: some output\n]\n"
            + "[STDERR: an error\n]\n",
            out.toString());
  }
}
