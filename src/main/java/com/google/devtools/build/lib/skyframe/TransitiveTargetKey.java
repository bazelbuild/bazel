// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * A key requesting transitive loading of all dependencies of a given label; see
 * {@link TransitiveTargetFunction} and {@link TransitiveTargetValue}.
 */
@Immutable
@ThreadSafe
public final class TransitiveTargetKey implements SkyKey {
  public static TransitiveTargetKey of(Label label) {
    Preconditions.checkArgument(!label.getRepository().isDefault());
    return new TransitiveTargetKey(label);
  }

  private final Label label;

  private TransitiveTargetKey(Label label) {
    this.label = Preconditions.checkNotNull(label);
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.TRANSITIVE_TARGET;
  }

  @Override
  public Object argument() {
    return this;
  }

  public Label getLabel() {
    return label;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("label", label).toString();
  }

  @Override
  public int hashCode() {
    return 31 * functionName().hashCode() + label.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (!(o instanceof TransitiveTargetKey)) {
      return false;
    }
    return ((TransitiveTargetKey) o).label.equals(label);
  }
}
