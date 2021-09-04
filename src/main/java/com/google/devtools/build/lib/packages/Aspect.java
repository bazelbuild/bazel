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

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

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

  /**
   * The aspect definition is a function of the aspect class + its parameters, so we can cache that.
   *
   * <p>The native aspects are loaded with blaze and are not stateful. Reference equality works fine
   * in this case.
   *
   * <p>Caching of Starlark aspects is not yet implemented.
   */
  private static final LoadingCache<
          NativeAspectClass, LoadingCache<AspectParameters, AspectDefinition>>
      definitionCache =
          Caffeine.newBuilder()
              .build(
                  nativeAspectClass ->
                      Caffeine.newBuilder().build(nativeAspectClass::getDefinition));

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

  private Aspect(
      AspectClass aspectClass,
      AspectDefinition aspectDefinition,
      AspectParameters parameters,
      RequiredProviders inheritedRequiredProviders,
      ImmutableSet<String> inheritedAttributeAspects) {
    this.aspectDescriptor =
        new AspectDescriptor(
            Preconditions.checkNotNull(aspectClass),
            Preconditions.checkNotNull(parameters),
            inheritedRequiredProviders,
            inheritedAttributeAspects);
    this.aspectDefinition = Preconditions.checkNotNull(aspectDefinition);
  }

  public static Aspect forNative(
      NativeAspectClass nativeAspectClass,
      AspectParameters parameters,
      RequiredProviders inheritedRequiredProviders,
      ImmutableSet<String> inheritedAttributeAspects) {
    AspectDefinition definition = definitionCache.get(nativeAspectClass).get(parameters);
    return new Aspect(
        nativeAspectClass,
        definition,
        parameters,
        inheritedRequiredProviders,
        inheritedAttributeAspects);
  }

  public static Aspect forNative(
      NativeAspectClass nativeAspectClass, AspectParameters parameters) {
    AspectDefinition definition = definitionCache.get(nativeAspectClass).get(parameters);
    return new Aspect(nativeAspectClass, definition, parameters);
  }

  public static Aspect forNative(NativeAspectClass nativeAspectClass) {
    return forNative(nativeAspectClass, AspectParameters.EMPTY);
  }

  public static Aspect forStarlark(
      StarlarkAspectClass starlarkAspectClass,
      AspectDefinition aspectDefinition,
      AspectParameters parameters,
      RequiredProviders inheritedRequiredProviders,
      ImmutableSet<String> inheritedAttributeAspects) {
    return new Aspect(
        starlarkAspectClass,
        aspectDefinition,
        parameters,
        inheritedRequiredProviders,
        inheritedAttributeAspects);
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

  /** {@link ObjectCodec} for {@link Aspect}. */
  @SuppressWarnings("unused") // Used reflectively.
  private static final class AspectCodec implements ObjectCodec<Aspect> {
    @Override
    public Class<Aspect> getEncodedClass() {
      return Aspect.class;
    }

    @Override
    public void serialize(SerializationContext context, Aspect obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.getDescriptor(), codedOut);
      boolean nativeAspect = obj.getDescriptor().getAspectClass() instanceof NativeAspectClass;
      codedOut.writeBoolNoTag(nativeAspect);
      if (!nativeAspect) {
        context.serialize(obj.getDefinition(), codedOut);
      }
    }

    @Override
    public Aspect deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      AspectDescriptor aspectDescriptor = context.deserialize(codedIn);
      if (codedIn.readBool()) {
        return forNative(
            (NativeAspectClass) aspectDescriptor.getAspectClass(),
            aspectDescriptor.getParameters());
      } else {
        AspectDefinition aspectDefinition = context.deserialize(codedIn);
        return forStarlark(
            (StarlarkAspectClass) aspectDescriptor.getAspectClass(),
            aspectDefinition,
            aspectDescriptor.getParameters(),
            aspectDescriptor.getInheritedRequiredProviders(),
            aspectDescriptor.getInheritedAttributeAspects());
      }
    }
  }
}
