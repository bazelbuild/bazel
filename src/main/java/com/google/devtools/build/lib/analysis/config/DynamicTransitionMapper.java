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
package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.packages.RuleClass;

/**
 * Maps non-{@link PatchTransition} declarations to their implementable equivalents.
 *
 * <p>Blaze applies configuration transitions by executing {@link PatchTransition} instances. But
 * for legacy reasons, not every transition declaration is a {@link PatchTransition}. The most
 * prominent example is {@link Attribute.ConfigurationTransition}, which defines its transitions as
 * enums. These transitions are used all over the place. So we need a way to continue to support
 * them.
 *
 * <p>Hence this class.
 *
 * <p>Going forward, we should eliminate the need for this class by eliminating
 * non-{@link PatchTransition} transitions. This is conceptually straightforward: replace
 * declarations of the form {@link RuleClass.Builder#cfg(Transition)} with
 * {@link RuleClass.Builder#cfg(PatchTransition)}. That way, transition declarations "just work",
 * with no extra fuss. But this is a migration that will take some time to complete.
 *
 * {@link Attribute.ConfigurationTransition#DATA} provides the most complicated challenge. This is
 * C++/LIPO logic, and the implementation is in C++ rule code
 * ({@link com.google.devtools.build.lib.rules.cpp.transitions.DisableLipoTransition}). But the enum
 * is defined in {@link Attribute}, which is in {@code lib.packages}, which has access to neither
 * rule-specific nor configuration-specific code. Furthermore, many non-C++ rules declare this
 * transition. We ultimately need a cleaner way to inject this rules-specific logic into general
 * Blaze code.
  */
public final class DynamicTransitionMapper {
  /**
   * Use this to declare a no-op transition that keeps the input configuration.
   */
  public static final Transition SELF = () -> {
      throw new UnsupportedOperationException("This is just an alias for \"keep the input "
       + "configuration\". It shouldn't actually apply a real transition");
  };

  private final ImmutableMap<Transition, Transition> map;

  /**
   * Creates a new mapper with the given mapping. Any transition not in this mapping triggers
   * an {@link IllegalArgumentException}.
   */
  public DynamicTransitionMapper(ImmutableMap<Transition, Transition> map) {
    this.map = map;
  }

  /**
   * Given an input transition, returns the equivalent transition Blaze's implementation logic knows
   * how to apply.
   *
   * <p>When the input is a {@link PatchTransition}, this just returns the input. This is because
   * that's the kind of transition that Blaze natively applies. For this reason, all inputs should
   * ideally be {@link PatchTransition}s.
   *
   * <p>Non-{@link PatchTransition} inputs that aren't mapped here throw an
   * {@link IllegalArgumentException}.
   */
  public Transition map(Transition fromTransition) {
    if (fromTransition instanceof PatchTransition || fromTransition == null) {
      return fromTransition;
    }
    Transition toTransition = map.get(fromTransition);
    if (toTransition == SELF) {
      return fromTransition;
    } else if (toTransition != null) {
      return toTransition;
    }
    throw new IllegalArgumentException("No dynamic mapping for " + fromTransition.toString());
  }
}
