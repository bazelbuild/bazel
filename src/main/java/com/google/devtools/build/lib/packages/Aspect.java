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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * An instance of a given {@code AspectClass} with loaded definition and parameters.
 *
 * This is an aspect equivalent of {@link Rule} class for build rules.
 *
 * Note: this class does not have {@code equals()} and {@code hashCode()} redefined, so should
 * not be used in SkyKeys.
 */
@Immutable
public final class Aspect implements DependencyFilter.AttributeInfoProvider {
  // TODO(bazel-team): class objects are not really hashable or comparable for equality other than
  // by reference. We should identify the aspect here in a way that does not rely on comparison
  // by reference so that keys can be serialized and deserialized properly.
  private final AspectClass aspectClass;
  private final AspectParameters parameters;
  private final AspectDefinition aspectDefinition;

  private Aspect(
      AspectClass aspectClass,
      AspectDefinition aspectDefinition,
      AspectParameters parameters) {
    this.aspectClass = Preconditions.checkNotNull(aspectClass);
    this.aspectDefinition = Preconditions.checkNotNull(aspectDefinition);
    this.parameters = Preconditions.checkNotNull(parameters);
  }

  public static Aspect forNative(
      NativeAspectClass nativeAspectClass, AspectParameters parameters) {
    return new Aspect(nativeAspectClass, nativeAspectClass.getDefinition(parameters), parameters);
  }

  public static Aspect forNative(NativeAspectClass nativeAspectClass) {
    return forNative(nativeAspectClass, AspectParameters.EMPTY);
  }

  public static Aspect forSkylark(
      SkylarkAspectClass skylarkAspectClass,
      AspectDefinition definition,
      AspectParameters parameters) {
    return new Aspect(skylarkAspectClass, definition, parameters);
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
  public String toString() {
    return String.format("Aspect %s(%s)", aspectClass, parameters);
  }

  public AspectDefinition getDefinition() {
    return aspectDefinition;
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    // All aspect attributes are implicit.
    return false;
  }
}
