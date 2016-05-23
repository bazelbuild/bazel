// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.server.signal.AbstractSignalHandler;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;

import sun.misc.Signal;

import java.util.logging.Logger;

/**
 * Class that causes Blaze to exit with {@link ExitCode#OOM_ERROR} when the JVM receives a
 * {@code SIGUSR2} signal.
 *
 * <p>The Blaze client can be configured to send {@code SIGUSR2} on an OOM.
 */
class OomSignalHandler extends AbstractSignalHandler {
  private static final Logger LOG = Logger.getLogger(OomSignalHandler.class.getName());
  private static final Signal SIGUSR2 = new Signal("USR2");
  private static final String MESSAGE = "SIGUSR2 received, presumably from JVM due to OOM";
  // Pre-allocate memory for object that will be used during an OOM, because there may not be spare
  // memory when we're OOMing. That's kind of the point of an OOM.
  private static final OutOfMemoryError OUT_OF_MEMORY_ERROR = new OutOfMemoryError(MESSAGE);

  OomSignalHandler() {
    super(SIGUSR2);
  }

  @Override
  protected void onSignal() {
    LOG.info(MESSAGE);
    OutErr.SYSTEM_OUT_ERR.printErrLn(
        "Exiting as if we OOM'd because SIGUSR2 received, presumably from JVM");
    BugReport.handleCrash(OUT_OF_MEMORY_ERROR);
  }
}
