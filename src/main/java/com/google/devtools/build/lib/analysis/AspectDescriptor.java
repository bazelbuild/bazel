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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectParameters;

import java.util.Objects;

/**
 * A pair of {@link AspectClass} and {@link AspectParameters}.
 *
 * Used for dependency resolution.
 */
@Immutable
public final class AspectDescriptor {
  private final AspectClass aspectClass;
  private final AspectParameters aspectParameters;

  public AspectDescriptor(AspectClass aspectClass,
      AspectParameters aspectParameters) {
    this.aspectClass = aspectClass;
    this.aspectParameters = aspectParameters;
  }

  public AspectDescriptor(AspectClass aspectClass) {
    this(aspectClass, AspectParameters.EMPTY);
  }

  public AspectClass getAspectClass() {
    return aspectClass;
  }

  public AspectParameters getParameters() {
    return aspectParameters;
  }

  @Override
  public int hashCode() {
    return Objects.hash(aspectClass, aspectParameters);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }

    if (!(obj instanceof AspectDescriptor)) {
      return false;
    }

    AspectDescriptor that = (AspectDescriptor) obj;
    return Objects.equals(aspectClass, that.aspectClass)
        && Objects.equals(aspectParameters, that.aspectParameters);
  }
}
