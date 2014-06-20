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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.syntax.Label;

/**
 * This exception gets thrown if there were errors during the execution phase of
 * the build.
 *
 * <p>The argument to the constructor may be null if the thrower has already
 * printed an error message; in this case, no error message should be printed by
 * the catcher. (Typically, this happens when the builder is unsuccessful and
 * {@code --keep_going} was specified. This error corresponds to one or more
 * actions failing, but since those actions' failures will be reported
 * separately, the exception carries no message and is just used for control
 * flow.)
 */
@ThreadSafe
public class BuildFailedException extends Exception {
  private final boolean catastrophic;
  private final Action action;
  private final Iterable<Label> rootCauses;
  private final boolean errorAlreadyShown;

  public BuildFailedException() {
    this(null);
  }

  public BuildFailedException(String message) {
    this(message, false, null, ImmutableList.<Label>of());
  }

  public BuildFailedException(String message, boolean catastrophic) {
    this(message, catastrophic, null, ImmutableList.<Label>of());
  }

  public BuildFailedException(String message, boolean catastrophic,
      Action action, Iterable<Label> rootCauses) {
    this(message, catastrophic, action, rootCauses, false);
  }

  public BuildFailedException(String message, boolean catastrophic,
      Action action, Iterable<Label> rootCauses, boolean errorAlreadyShown) {
    super(message);
    this.catastrophic = catastrophic;
    this.rootCauses = ImmutableList.copyOf(rootCauses);
    this.action = action;
    this.errorAlreadyShown = errorAlreadyShown;
  }

  public boolean isCatastrophic() {
    return catastrophic;
  }

  public Action getAction() {
    return action;
  }

  public Iterable<Label> getRootCauses() {
    return rootCauses;
  }

  public boolean isErrorAlreadyShown() {
    return errorAlreadyShown || getMessage() == null;
  }
}
