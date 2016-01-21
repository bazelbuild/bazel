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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.util.Preconditions;

import java.util.Objects;

/**
 * Wrapper around {@link AspectClass} class and {@link AspectParameters}. Aspects are
 * created with help of aspect factory instances and parameters are used to configure them, so we
 * have to keep them together.
 */
public final class Aspect implements DependencyFilter.AttributeInfoProvider {
  // TODO(bazel-team): class objects are not really hashable or comparable for equality other than
  // by reference. We should identify the aspect here in a way that does not rely on comparison
  // by reference so that keys can be serialized and deserialized properly.
  private final AspectClass aspectClass;
  private final AspectParameters parameters;

  public Aspect(AspectClass aspect, AspectParameters parameters) {
    Preconditions.checkNotNull(aspect);
    Preconditions.checkNotNull(parameters);
    this.aspectClass = aspect;
    this.parameters = parameters;
  }

  public Aspect(AspectClass aspect) {
    this(aspect, AspectParameters.EMPTY);
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
    if (!(other instanceof Aspect)) {
      return false;
    }
    Aspect that = (Aspect) other;
    return Objects.equals(this.aspectClass, that.aspectClass)
        && Objects.equals(this.parameters, that.parameters);
  }

  @Override
  public int hashCode() {
    return Objects.hash(aspectClass, parameters);
  }

  @Override
  public String toString() {
    return String.format("Aspect %s(%s)", aspectClass, parameters);
  }

  public AspectDefinition getDefinition() {
    return aspectClass.getDefinition(parameters);
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    // All aspect attributes are implicit.
    return false;
  }
}
