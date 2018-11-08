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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import java.io.Serializable;
import java.util.Set;
import java.util.concurrent.ExecutionException;

/** An identifier for a {@code SkyFunction}. */
public final class SkyFunctionName implements Serializable {
  private static final Cache<NameOnlyWrapper, SkyFunctionName> interner =
      CacheBuilder.newBuilder().build();

  /**
   * A well-known key type intended for testing only. The associated SkyKey should have a String
   * argument.
   */
  // Needs to be after the cache is initialized.
  public static final SkyFunctionName FOR_TESTING = SkyFunctionName.createHermetic("FOR_TESTING");

  /**
   * Creates a SkyFunctionName identified by {@code name} whose evaluation is non-hermetic (its
   * value may not be a pure function of its dependencies. Only use this if the evaluation
   * explicitly consumes data outside of Skyframe, or if the node can be directly invalidated (as
   * opposed to transitively invalidated).
   */
  public static SkyFunctionName createNonHermetic(String name) {
    return create(name, ShareabilityOfValue.SOMETIMES, FunctionHermeticity.NONHERMETIC);
  }

  /**
   * Creates a SkyFunctionName identified by {@code name} whose evaluation is {@linkplain
   * FunctionHermeticity#SEMI_HERMETIC semi-hermetic}.
   */
  public static SkyFunctionName createSemiHermetic(String name) {
    return create(name, ShareabilityOfValue.SOMETIMES, FunctionHermeticity.SEMI_HERMETIC);
  }

  /**
   * Creates a SkyFunctionName identified by {@code name} whose evaluation is hermetic (guaranteed
   * to be a deterministic function of its dependencies, not doing any external operations).
   */
  public static SkyFunctionName createHermetic(String name) {
    return create(name, ShareabilityOfValue.SOMETIMES, FunctionHermeticity.HERMETIC);
  }

  public static SkyFunctionName create(
      String name, ShareabilityOfValue shareabilityOfValue, FunctionHermeticity hermeticity) {
    SkyFunctionName result = new SkyFunctionName(name, shareabilityOfValue, hermeticity);
    SkyFunctionName cached;
    try {
      cached = interner.get(new NameOnlyWrapper(result), () -> result);
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    }
    Preconditions.checkState(
        result.equals(cached),
        "Tried to create SkyFunctionName objects with same name but different properties: %s %s",
        result,
        cached);
    return cached;
  }

  private final String name;
  private final ShareabilityOfValue shareabilityOfValue;
  private final FunctionHermeticity hermeticity;

  private SkyFunctionName(
      String name, ShareabilityOfValue shareabilityOfValue, FunctionHermeticity hermeticity) {
    this.name = name;
    this.shareabilityOfValue = shareabilityOfValue;
    this.hermeticity = hermeticity;
  }

  public String getName() {
    return name;
  }

  public ShareabilityOfValue getShareabilityOfValue() {
    return shareabilityOfValue;
  }

  public FunctionHermeticity getHermeticity() {
    return hermeticity;
  }

  @Override
  public String toString() {
    return name + (shareabilityOfValue.equals(ShareabilityOfValue.NEVER) ? " (unshareable)" : "");
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof SkyFunctionName)) {
      return false;
    }
    SkyFunctionName other = (SkyFunctionName) obj;
    return name.equals(other.name) && shareabilityOfValue.equals(other.shareabilityOfValue);
  }

  @Override
  public int hashCode() {
    // Don't bother incorporating serializabilityOfValue into hashCode: should always be the same.
    return name.hashCode();
  }

  /**
   * A predicate that returns true for {@link SkyKey}s that have the given {@link SkyFunctionName}.
   */
  public static Predicate<SkyKey> functionIs(final SkyFunctionName functionName) {
    return skyKey -> functionName.equals(skyKey.functionName());
  }

  /**
   * A predicate that returns true for {@link SkyKey}s that have the given {@link SkyFunctionName}.
   */
  public static Predicate<SkyKey> functionIsIn(final Set<SkyFunctionName> functionNames) {
    return skyKey -> functionNames.contains(skyKey.functionName());
  }

  private static class NameOnlyWrapper {
    private final SkyFunctionName skyFunctionName;

    private NameOnlyWrapper(SkyFunctionName skyFunctionName) {
      this.skyFunctionName = skyFunctionName;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof NameOnlyWrapper)) {
        return false;
      }
      SkyFunctionName thatFunctionName = ((NameOnlyWrapper) obj).skyFunctionName;
      return this.skyFunctionName.getName().equals(thatFunctionName.name);
    }

    @Override
    public int hashCode() {
      return skyFunctionName.getName().hashCode();
    }
  }
}
