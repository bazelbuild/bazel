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
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransitionProxy;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.Transition;

/**
 * Maps non-{@link PatchTransition} declarations to their implementable equivalents.
 *
 * <p>Blaze applies configuration transitions by executing {@link PatchTransition} instances. But
 * for legacy reasons, {@link ConfigurationTransitionProxy#DATA} (which is used for C++/LIPO logic)
 * is an implementation-free enum.
 *
 * <p>We should ultimately restrict that logic just to the C++ rule definitions and remove this
 * interface. But {@link ConfigurationTransitionProxy#DATA} is used everywhere, including in
 * non-C++ rules and in {@code lib.packages} code, which lacks acccess to C++ configuration logic.
  */
public final class DynamicTransitionMapper {
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
    if (fromTransition instanceof PatchTransition
        || fromTransition instanceof SplitTransition
        || fromTransition == null) {
      return fromTransition;
    }
    Transition toTransition = map.get(fromTransition);
    if (toTransition == null) {
      throw new IllegalArgumentException("No dynamic mapping for " + fromTransition.toString());
    }
    return toTransition;
  }
}
