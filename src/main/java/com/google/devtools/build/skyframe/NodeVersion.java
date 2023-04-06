// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.auto.value.AutoValue;

/**
 * Encapsulates the two versions relevant to a {@link NodeEntry}: when it was last evaluated, and
 * when its value last changed.
 */
public interface NodeVersion {

  /**
   * Returns the last version at which a node's value changed.
   *
   * <p>In {@link NodeEntry#setValue} it may be determined that the value being set is the same as
   * the already-stored value. In that case, the last changed version will remain the same.
   */
  Version lastChanged();

  /**
   * Returns the last version a {@link NodeEntry} was evaluated at, even if it re-evaluated to the
   * same value.
   *
   * <p>When a child signals a node with the last version it was changed at in {@link
   * NodeEntry#signalDep}, the node need not re-evaluate if the child's version is {@link
   * Version#atMost} this version, even if {@link #lastChanged} is lower.
   */
  Version lastEvaluated();

  static NodeVersion of(Version lastChanged, Version lastEvaluated) {
    if (lastChanged.equals(lastEvaluated)) {
      return lastChanged;
    }
    return new AutoValue_NodeVersion_ChangePruned(lastChanged, lastEvaluated);
  }

  /**
   * Basic implementation of {@link NodeVersion} for the case where {@link #lastChanged} and {@link
   * #lastEvaluated} are different versions.
   */
  @AutoValue
  abstract class ChangePruned implements NodeVersion {}
}
