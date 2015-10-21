// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectParameters;

import java.util.Objects;

/**
 * Wrapper around {@link ConfiguredAspectFactory} class and {@link AspectParameters}. Aspects are
 * created with help of aspect factory instances and parameters are used to configure them, so we
 * have to keep them together. 
 */
public final class AspectWithParameters {
  // TODO(bazel-team): class objects are not really hashable or comparable for equality other than
  // by reference. We should identify the aspect here in a way that does not rely on comparison
  // by reference so that keys can be serialized and deserialized properly.
  private final AspectClass aspectClass;
  private final AspectParameters parameters;

  public AspectWithParameters(AspectClass aspect, AspectParameters parameters) {
    Preconditions.checkNotNull(parameters);
    this.aspectClass = aspect;
    this.parameters = parameters;
  }

  public AspectWithParameters(AspectClass aspect) {
    this.aspectClass = aspect;
    this.parameters = AspectParameters.EMPTY;
  }

  /**
   * Returns the aspectClass required for building the aspect.
   */
  public AspectClass getAspectClass() {
    return aspectClass;
  }

  /**
   * Returns parameters for evaluation of the aspect.
   */
  public AspectParameters getParameters() {
    return parameters;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof AspectWithParameters)) {
      return false;
    }
    AspectWithParameters that = (AspectWithParameters) other;
    return Objects.equals(this.aspectClass, that.aspectClass)
        && Objects.equals(this.parameters, that.parameters);
  }

  @Override
  public int hashCode() {
    return Objects.hash(aspectClass, parameters);
  }

  @Override
  public String toString() {
    return String.format("AspectWithParameters %s(%s)", aspectClass, parameters);
  }
}
