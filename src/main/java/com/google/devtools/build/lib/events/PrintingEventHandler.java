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
package com.google.devtools.build.lib.events;

import com.google.devtools.build.lib.util.io.OutErr;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Set;

/**
 * An event handler that prints to an OutErr stream pair in a
 * canonical format, for example:
 * <pre>
 * ERROR: /home/jrluser/src/workspace/x/BUILD:23:1: syntax error.
 * </pre>
 * This syntax is parseable by Emacs's compile.el.
 *
 * <p>
 * By default, the output will go to SYSTEM_OUT_ERR,
 * but this can be changed by calling the setOut() method.
 *
 * <p>
 * This class is used only for tests.
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

  private OutErr outErr = OutErr.SYSTEM_OUT_ERR;

  /**
   * Setup a printing event handler that will handle events matching the mask.
   * Events will be printed to the original System.out and System.err
   * unless/until redirected by a call to setOutErr().
   */
  public PrintingEventHandler(Set<EventKind> mask) {
    super(mask);
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
          PrintWriter err = new PrintWriter(outErr.getErrorStream());
          err.print(event.getKind());
          err.print(": ");
          if (event.getLocation() != null) {
            err.print(event.getLocation().print());
            err.print(": ");
          }
          err.println(event.getMessage());
          err.flush();
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
  }
}
