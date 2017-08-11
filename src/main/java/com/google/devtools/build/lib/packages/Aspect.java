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
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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

  /** */
  public static final String INJECTING_RULE_KIND_PARAMETER_KEY = "$injecting_rule_kind";

  /**
   * The aspect definition is a function of the aspect class + its parameters, so we can cache that.
   *
   * <p>The native aspects are loaded with blaze and are not stateful. Reference equality works fine
   * in this case.
   *
   * <p>Caching of Skylark aspects is not yet implemented.
   */
  private static final Map<NativeAspectClass, Map<AspectParameters, AspectDefinition>>
      definitionCache = new ConcurrentHashMap<>();

  private final AspectDescriptor aspectDescriptor;
  private final AspectDefinition aspectDefinition;

  private Aspect(
      AspectClass aspectClass,
      AspectDefinition aspectDefinition,
      AspectParameters parameters) {
    this.aspectDescriptor = new AspectDescriptor(
        Preconditions.checkNotNull(aspectClass),
        Preconditions.checkNotNull(parameters));
    this.aspectDefinition = Preconditions.checkNotNull(aspectDefinition);
  }

  public static Aspect forNative(
      NativeAspectClass nativeAspectClass, AspectParameters parameters) {
    AspectDefinition definition =
        definitionCache
            .computeIfAbsent(nativeAspectClass, key -> new ConcurrentHashMap<>())
            .computeIfAbsent(parameters, nativeAspectClass::getDefinition);
    return new Aspect(nativeAspectClass, definition, parameters);
  }

  public static Aspect forNative(NativeAspectClass nativeAspectClass) {
    return forNative(nativeAspectClass, AspectParameters.EMPTY);
  }

  public static Aspect forSkylark(
      SkylarkAspectClass skylarkAspectClass,
      AspectDefinition aspectDefinition,
      AspectParameters parameters) {
    return new Aspect(skylarkAspectClass, aspectDefinition, parameters);
  }

  /**
   * Returns the aspectClass required for building the aspect.
   */
  public AspectClass getAspectClass() {
    return aspectDescriptor.getAspectClass();
  }

  /**
   * Returns parameters for evaluation of the aspect.
   */
  public AspectParameters getParameters() {
    return aspectDescriptor.getParameters();
  }

  public AspectDescriptor getDescriptor() {
    return aspectDescriptor;
  }

  @Override
  public String toString() {
    return String.format("Aspect %s", aspectDescriptor.toString());
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
