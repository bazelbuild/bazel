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

package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ExecException;

/**
 * An ExecException which happens because we were interrupted while attempting
 * something.
 */
public class InterruptedExecException extends ExecException {

  public InterruptedExecException(String message) {
    super(message, false);
  }

  public InterruptedExecException(String message, Throwable cause) {
    super(message, cause, false);
  }

  @Override
  public ActionExecutionException toActionExecutionException(String messagePrefix,
      boolean verboseFailures, Action action) {
    return new ActionExecutionException(messagePrefix + " interrupted: " + getMessage(),
        getCause(), action, true);
  }
}
