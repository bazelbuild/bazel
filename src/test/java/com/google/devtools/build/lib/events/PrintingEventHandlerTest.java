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
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests the {@link PrintingEventHandler}.
 */
@RunWith(JUnit4.class)
public class PrintingEventHandlerTest extends EventTestTemplate {

  @Test
  public void collectsEvents() {
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    PrintingEventHandler handler =
        new PrintingEventHandler(recordingOutErr, EventKind.ERRORS_AND_WARNINGS);
    handler.handle(event);
    MoreAsserts.assertEqualsUnifyingLineEnds("WARNING: /path/to/workspace/my/sample/path.txt:3:4: "
                 + "This is not an error message.\n",
                 recordingOutErr.errAsLatin1());
    assertThat(recordingOutErr.outAsLatin1()).isEmpty();
  }

  @Test
  public void nonAsciiLocationAndMessages() {
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    PrintingEventHandler handler = new PrintingEventHandler(recordingOutErr, EventKind.ALL_EVENTS);
    Location location = Location.fromFileLineColumn(unicodeToInternal("/wörkspace/BUILD"), 3, 4);

    handler.handle(Event.error(location, unicodeToInternal("internal: Grüße 🌱")));
    handler.handle(Event.error(location, "not internal: ☃"));
    handler.handle(Event.of(EventKind.STDERR, "stderr: ☃\n"));
    handler.handle(Event.of(EventKind.STDOUT, unicodeToInternal("stdout: Grüße 🌱\n")));

    assertThat(recordingOutErr.getErrorStream().toString(UTF_8))
        .isEqualTo(
            """
            ERROR: /wörkspace/BUILD:3:4: internal: Grüße 🌱
            ERROR: /wörkspace/BUILD:3:4: not internal: ☃
            stderr: ☃
            """);
    assertThat(recordingOutErr.getOutputStream().toString(UTF_8)).isEqualTo("stdout: Grüße 🌱\n");
  }
}
