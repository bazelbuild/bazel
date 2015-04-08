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
package com.google.devtools.build.lib.actions;

/**
 * Methods needed by {@code SkyframeBuilder}.
 */
public final class BuilderUtils {

  private BuilderUtils() {}

  /**
   * Figure out why an action's execution failed and rethrow the right kind of exception.
   */
  public static void rethrow(Throwable cause) throws BuildFailedException, TestExecException {
    Throwable innerCause = cause.getCause();
    if (innerCause instanceof TestExecException) {
      throw (TestExecException) innerCause;
    }
    if (cause instanceof ActionExecutionException) {
      ActionExecutionException actionExecutionCause = (ActionExecutionException) cause;
      // Sometimes ActionExecutionExceptions are caused by Actions with no owner.
      String message =
          (actionExecutionCause.getLocation() != null) ?
          (actionExecutionCause.getLocation().print() + " " + cause.getMessage()) :
          cause.getMessage();
      throw new BuildFailedException(message, actionExecutionCause.isCatastrophe(),
          actionExecutionCause.getAction(), actionExecutionCause.getRootCauses(),
          /*errorAlreadyShown=*/ !actionExecutionCause.showError());
    } else if (cause instanceof MissingInputFileException) {
      throw new BuildFailedException(cause.getMessage());
    } else if (cause instanceof RuntimeException) {
      throw (RuntimeException) cause;
    } else if (cause instanceof Error) {
      throw (Error) cause;
    } else {
      // We encountered an exception we don't think we should have encountered. This can indicate
      // a bug in our code, such as lower level exceptions not being properly handled, or in our
      // expectations in this method.
      throw new IllegalArgumentException("action terminated with "
          + "unexpected exception: " + cause.getMessage(), cause);
    }
  }
}
