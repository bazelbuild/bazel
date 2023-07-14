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

import static com.google.common.base.Preconditions.checkNotNull;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * An instance of a given {@code AspectClass} with loaded definition and parameters.
 *
 * <p>This is an aspect equivalent of {@link Rule} class for build rules.
 *
 * <p>Note: equality is only implemented for purposes of interning. It delegates to {@link
 * AspectDefinition} equality, which is not overridden. For this reason, this class should not be
 * used in SkyKeys - use {@link AspectDescriptor} instead.
 */
@Immutable
public final class Aspect implements DependencyFilter.AttributeInfoProvider {

  /**
   * The aspect definition is a function of the aspect class + its parameters, so we can cache that.
   *
   * <p>The native aspects are loaded with blaze and are not stateful. Reference equality works fine
   * in this case.
   */
  private static final LoadingCache<
          NativeAspectClass, LoadingCache<AspectParameters, AspectDefinition>>
      definitionCache =
          Caffeine.newBuilder()
              .build(
                  nativeAspectClass ->
                      Caffeine.newBuilder().build(nativeAspectClass::getDefinition));

  private static final Interner<Aspect> interner = BlazeInterners.newWeakInterner();

  private final AspectDescriptor aspectDescriptor;
  private final AspectDefinition aspectDefinition;

  private Aspect(AspectDescriptor aspectDescriptor, AspectDefinition aspectDefinition) {
    this.aspectDescriptor = checkNotNull(aspectDescriptor);
    this.aspectDefinition = checkNotNull(aspectDefinition);
  }

  public static Aspect forNative(NativeAspectClass nativeAspectClass, AspectParameters parameters) {
    AspectDefinition definition = definitionCache.get(nativeAspectClass).get(parameters);
    return createInterned(nativeAspectClass, definition, parameters);
  }

  public static Aspect forNative(NativeAspectClass nativeAspectClass) {
    return forNative(nativeAspectClass, AspectParameters.EMPTY);
  }

  public static Aspect forStarlark(
      StarlarkAspectClass starlarkAspectClass,
      AspectDefinition aspectDefinition,
      AspectParameters parameters) {
    return createInterned(starlarkAspectClass, aspectDefinition, parameters);
  }

  private static Aspect createInterned(
      AspectClass aspectClass, AspectDefinition definition, AspectParameters parameters) {
    return interner.intern(new Aspect(AspectDescriptor.of(aspectClass, parameters), definition));
  }

  /** Returns the aspectClass required for building the aspect. */
  public AspectClass getAspectClass() {
    return aspectDescriptor.getAspectClass();
  }

  /** Returns parameters for evaluation of the aspect. */
  public AspectParameters getParameters() {
    return aspectDescriptor.getParameters();
  }

  public AspectDescriptor getDescriptor() {
    return aspectDescriptor;
  }

  public AspectDefinition getDefinition() {
    return aspectDefinition;
  }

  @Override
  public boolean isAttributeValueExplicitlySpecified(Attribute attribute) {
    // All aspect attributes are implicit.
    return false;
  }

  @Override
  public String toString() {
    return "Aspect " + aspectDescriptor;
  }

  @Override
  public int hashCode() {
    return HashCodes.hashObjects(aspectDescriptor, aspectDefinition);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof Aspect)) {
      return false;
    }

    Aspect that = (Aspect) obj;
    return aspectDescriptor.equals(that.aspectDescriptor)
        && aspectDefinition.equals(that.aspectDefinition);
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
            aspectDescriptor.getParameters());
      }
    }
  }
}
