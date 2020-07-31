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

import com.google.devtools.build.lib.util.io.OutErr;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Set;

/**
 * An event handler that prints to an OutErr stream pair in a
 * canonical format, for example:
 * <pre>
 * ERROR: /home/jrluser/src/workspace/x/BUILD:23:1: syntax error.
 * </pre>
 * This syntax is parseable by Emacs's compile.el.
 */
public class PrintingEventHandler extends AbstractEventHandler
    implements EventHandler {

  /**
   * A convenient event-handler for terminal applications that prints all
   * errors and warnings it encounters to the error stream.
   * STDOUT and STDERR events pass their output directly
   * through to the corresponding streams.
   */
  public static final PrintingEventHandler ERRORS_AND_WARNINGS_TO_STDERR =
      new PrintingEventHandler(EventKind.ERRORS_AND_WARNINGS_AND_OUTPUT);

  /**
   * A convenient event-handler for terminal applications that prints all
   * errors it encounters to the error stream.
   * STDOUT and STDERR events pass their output directly
   * through to the corresponding streams.
   */
  public static final PrintingEventHandler ERRORS_TO_STDERR =
      new PrintingEventHandler(EventKind.ERRORS_AND_OUTPUT);

  private OutErr outErr;

  /**
   * Setup a printing event handler that prints events matching the mask.
   */
  public PrintingEventHandler(OutErr outErr, Set<EventKind> mask) {
    super(mask);
    this.outErr = outErr;
  }

  /**
   * Setup a printing event handler that prints events matching the mask. Events are printed to the
   * System.out and System.err unless/until redirected by a call to setOutErr().
   */
  public PrintingEventHandler(Set<EventKind> mask) {
    this(OutErr.SYSTEM_OUT_ERR, mask);
  }

  /**
   * Redirect all output to the specified OutErr stream pair.
   * Returns the previous OutErr.
   */
  public OutErr setOutErr(OutErr outErr) {
    OutErr prev = this.outErr;
    this.outErr = outErr;
    return prev;
  }

  /**
   * Print a description of the specified event to the appropriate
   * output or error stream.
   */
  @Override
  public void handle(Event event) {
    if (!getEventMask().contains(event.getKind())) {
      handleFollowUpEvents(event);
      return;
    }
    try {
      switch (event.getKind()) {
        case STDOUT:
          outErr.getOutputStream().write(event.getMessageBytes());
          outErr.getOutputStream().flush();
          break;
        case STDERR:
          outErr.getErrorStream().write(event.getMessageBytes());
          outErr.getErrorStream().flush();
          break;
        default:
          StringBuilder builder = new StringBuilder();
          builder.append(event.getKind()).append(": ");
          if (event.getLocation() != null) {
            builder.append(event.getLocation()).append(": ");
          }
          builder.append(event.getMessage()).append("\n");
          outErr.getErrorStream().write(builder.toString().getBytes(StandardCharsets.UTF_8));
          outErr.getErrorStream().flush();
      }
    } catch (IOException e) {
      /*
       * Note: we can't print to System.out or System.err here,
       * because those will normally be set to streams which
       * translate I/O to STDOUT and STDERR events,
       * which would result in infinite recursion.
       */
      outErr.printErrLn(e.getMessage());
    }
    handleFollowUpEvents(event);
  }

  private void handleFollowUpEvents(Event event) {
    byte[] stderr = event.getStdErr();
    if (stderr != null) {
      handle(Event.of(EventKind.STDERR, null, stderr));
    }
    byte[] stdout = event.getStdOut();
    if (stdout != null) {
      handle(Event.of(EventKind.STDOUT, null, stdout));
    }
  }
}
