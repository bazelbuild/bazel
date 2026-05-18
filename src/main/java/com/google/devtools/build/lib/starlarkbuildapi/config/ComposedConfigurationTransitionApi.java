// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.config;

import com.google.common.collect.ImmutableList;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A {@link ConfigurationTransitionApi} formed by composing transitions with {@code
 * transition.and_then}.
 *
 * <p>The elements are applied in order: the first reads the original configuration and each
 * subsequent transition reads the build settings produced by the previous one. Nested compositions
 * are flattened, so {@link #getElements} is always a flat chain of non-composed transitions.
 */
public final class ComposedConfigurationTransitionApi implements ConfigurationTransitionApi {

  private final ImmutableList<ConfigurationTransitionApi> elements;

  private ComposedConfigurationTransitionApi(ImmutableList<ConfigurationTransitionApi> elements) {
    this.elements = elements;
  }

  static ConfigurationTransitionApi compose(
      ConfigurationTransitionApi first, ConfigurationTransitionApi second) {
    ImmutableList.Builder<ConfigurationTransitionApi> elements = ImmutableList.builder();
    flatten(first, elements);
    flatten(second, elements);
    return new ComposedConfigurationTransitionApi(elements.build());
  }

  private static void flatten(
      ConfigurationTransitionApi transition,
      ImmutableList.Builder<ConfigurationTransitionApi> elements) {
    if (transition instanceof ComposedConfigurationTransitionApi composed) {
      elements.addAll(composed.elements);
    } else {
      elements.add(transition);
    }
  }

  /** Returns the flattened chain of transitions, in application order. */
  public ImmutableList<ConfigurationTransitionApi> getElements() {
    return elements;
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    printer.append("<transition object>");
  }
}
