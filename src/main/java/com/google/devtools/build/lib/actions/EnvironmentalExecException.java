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
import com.google.devtools.build.lib.util.ExitCode;
import java.io.IOException;

/**
 * An ExecException which is reports an issue executing an action due to an external problem on the
 * local system.
 *
 * <p>This exception will result in an exit code regarded as a system error; avoid using this for
 * problems which should be attributed to the user, e.g., a misconfigured BUILD file (use {@link
 * UserExecException}) or an action returning a non-zero exit code (use {@link *
 * com.google.devtools.build.lib.exec.SpawnExecException}.
 *
 * <p>The most common use of this exception is to wrap an "unexpected" {@link IOException} thrown by
 * an lower-level file system access or local process execution, e.g., failure to create a temporary
 * directory or denied file system access.
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
      return new ActionExecutionException(
          message, action, isCatastrophic(), ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    } else {
      String message = messagePrefix + " failed due to " + getMessage();
      return new ActionExecutionException(
          message, action, isCatastrophic(), ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    }
  }
}
