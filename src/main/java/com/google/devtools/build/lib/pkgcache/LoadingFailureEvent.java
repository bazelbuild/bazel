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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Objects;

/**
 * This event is fired during the build, when it becomes known that the loading
 * of a target cannot be completed because of an error in one of its
 * dependencies.
 */
public class LoadingFailureEvent {
  private final Label failedTarget;
  private final Label failureReason;

  public LoadingFailureEvent(Label failedTarget, Label failureReason) {
    this.failedTarget = failedTarget;
    this.failureReason = failureReason;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("failedTarget", failedTarget)
        .add("failureReason", failureReason)
        .toString();
  }

  public Label getFailedTarget() {
    return failedTarget;
  }

  public Label getFailureReason() {
    return failureReason;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    } else if (!(o instanceof LoadingFailureEvent)) {
      return false;
    }
    LoadingFailureEvent a = (LoadingFailureEvent) o;
    return Objects.equals(failedTarget, a.failedTarget)
        && Objects.equals(failureReason, a.failureReason);
  }

  @Override
  public int hashCode() {
    return Objects.hash(failedTarget, failureReason);
  }
}
