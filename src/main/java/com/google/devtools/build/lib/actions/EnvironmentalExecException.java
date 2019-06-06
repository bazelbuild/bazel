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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Throwables;
import java.io.IOException;

/**
 * An ExecException which is results from an external problem on the user's
 * local system.
 *
 * <p>Note that this is fundamentally different exception then the higher level
 * LocalEnvironmentException, which is thrown from the BuildTool. That exception
 * is thrown when the higher levels of Blaze decide to exit.
 *
 * <p>This exception is thrown when a low level error is encountered in the
 * strategy or client protocol layers.  This does not necessarily mean we will
 * exit; we may just retry the action.
 */
public class EnvironmentalExecException extends ExecException {
  public EnvironmentalExecException(IOException cause) {
    super("unexpected I/O exception", cause);
  }

  public EnvironmentalExecException(String message, Throwable cause) {
    super(message, cause);
  }

  public EnvironmentalExecException(String message) {
    super(message);
  }

  @Override
  public ActionExecutionException toActionExecutionException(
      String messagePrefix, boolean verboseFailures, Action action) {
    if (getCause() != null) {
      String message =
          messagePrefix
              + " failed due to "
              + getMessage()
              + "\n"
              + Throwables.getStackTraceAsString(getCause());
      return new ActionExecutionException(message, action, isCatastrophic());
    } else {
      String message = messagePrefix + " failed due to " + getMessage();
      return new ActionExecutionException(message, action, isCatastrophic());
    }
  }
}
