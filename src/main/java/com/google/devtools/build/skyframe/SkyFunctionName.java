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
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import java.util.Set;

/** An identifier for a {@code SkyFunction}. */
public final class SkyFunctionName {

  private static final Interner<SkyFunctionName> interner = BlazeInterners.newStrongInterner();

  /**
   * A well-known key type intended for testing only. The associated SkyKey should have a String
   * argument.
   */
  // Needs to be after the interner is initialized.
  public static final SkyFunctionName FOR_TESTING = SkyFunctionName.createHermetic("FOR_TESTING");

  /**
   * Creates a SkyFunctionName identified by {@code name} whose evaluation is non-hermetic (its
   * value may not be a pure function of its dependencies. Only use this if the evaluation
   * explicitly consumes data outside of Skyframe, or if the node can be directly invalidated (as
   * opposed to transitively invalidated).
   */
  public static SkyFunctionName createNonHermetic(String name) {
    return create(name, FunctionHermeticity.NONHERMETIC);
  }

  /**
   * Creates a SkyFunctionName identified by {@code name} whose evaluation is {@linkplain
   * FunctionHermeticity#SEMI_HERMETIC semi-hermetic}.
   */
  public static SkyFunctionName createSemiHermetic(String name) {
    return create(name, FunctionHermeticity.SEMI_HERMETIC);
  }

  /**
   * Creates a SkyFunctionName identified by {@code name} whose evaluation is hermetic (guaranteed
   * to be a deterministic function of its dependencies, not doing any external operations).
   */
  public static SkyFunctionName createHermetic(String name) {
    return create(name, FunctionHermeticity.HERMETIC);
  }

  private static SkyFunctionName create(String name, FunctionHermeticity hermeticity) {
    SkyFunctionName cached = interner.intern(new SkyFunctionName(name, hermeticity));
    Preconditions.checkState(
        cached.hermeticity.equals(hermeticity),
        "Tried to create SkyFunctionName objects with same name (%s) but different hermeticity"
            + " (old=%s, new=%s)",
        name,
        cached.hermeticity,
        hermeticity);
    return cached;
  }

  private final String name;
  private final FunctionHermeticity hermeticity;

  private SkyFunctionName(String name, FunctionHermeticity hermeticity) {
    this.name = Preconditions.checkNotNull(name);
    this.hermeticity = Preconditions.checkNotNull(hermeticity);
  }

  public String getName() {
    return name;
  }

  public FunctionHermeticity getHermeticity() {
    return hermeticity;
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof SkyFunctionName other)) {
      return false;
    }
    return name.equals(other.name);
  }

  @Override
  public int hashCode() {
    // Don't bother incorporating hermeticity into hashCode: should always be the same.
    return name.hashCode();
  }

  /**
   * A predicate that returns true for {@link SkyKey}s that have the given {@link SkyFunctionName}.
   */
  public static Predicate<SkyKey> functionIs(SkyFunctionName functionName) {
    return skyKey -> functionName.equals(skyKey.functionName());
  }

  /**
   * A predicate that returns true for {@link SkyKey}s that have the given {@link SkyFunctionName}.
   */
  public static Predicate<SkyKey> functionIsIn(Set<SkyFunctionName> functionNames) {
    return skyKey -> functionNames.contains(skyKey.functionName());
  }
}
