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

/**
 * An ExecException that is related to the failure of an Action and therefore
 * very likely the user's fault.
 */
public class UserExecException extends ExecException {

  public UserExecException(String message) {
    super(message);
  }
  
  public UserExecException(Throwable cause) {
    super(cause);
  }

  public UserExecException(String message, boolean timedOut) {
    super(message, false, timedOut);
  }

  public UserExecException(String message, Throwable cause) {
    super(message, cause);
  }

  public UserExecException(String message, Throwable cause, boolean timedOut) {
    super(message, cause, false, timedOut);
  }

  @Override
  public ActionExecutionException toActionExecutionException(String messagePrefix,
        boolean verboseFailures, Action action) {
    String message = messagePrefix + " failed";
    return new ActionExecutionException(message, this, action, isCatastrophic());
  }
}
