// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory.IncompatibleTransitionsException;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.starlarkbuildapi.config.ComposedConfigurationTransition;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigurationTransitionApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * Helpers for turning a {@link ComposedConfigurationTransition} (the deferred result of {@code
 * transition.and_then}) into a single {@link TransitionFactory} for a specific use context (rule
 * vs. attribute).
 */
final class ComposedTransitionMaterializer {

  private ComposedTransitionMaterializer() {}

  /** Converts one element of a composed transition into a {@link TransitionFactory}. */
  @FunctionalInterface
  interface ElementConverter<T extends TransitionFactory.Data> {
    TransitionFactory<T> convert(ConfigurationTransitionApi element) throws EvalException;
  }

  /**
   * Folds {@code composition}'s elements into a single {@link TransitionFactory} by converting each
   * with {@code converter} and combining them with {@link ComposingTransitionFactory#of}.
   *
   * @param incompatibleElementMessage error suffix used when an element can't be converted in this
   *     context (e.g. a native attribute-only transition used as a rule {@code cfg})
   */
  static <T extends TransitionFactory.Data> TransitionFactory<T> fold(
      ComposedConfigurationTransition composition,
      ElementConverter<T> converter,
      String incompatibleElementMessage)
      throws EvalException {
    TransitionFactory<T> result = null;
    for (ConfigurationTransitionApi element : composition.getElements()) {
      TransitionFactory<T> factory;
      try {
        factory = converter.convert(element);
      } catch (EvalException unused) {
        throw Starlark.errorf(
            "invalid composed transition for `cfg`: %s (composed at %s)",
            incompatibleElementMessage, composition.getLocation());
      }
      if (result == null) {
        result = factory;
      } else {
        try {
          result = ComposingTransitionFactory.of(result, factory);
        } catch (IncompatibleTransitionsException e) {
          throw Starlark.errorf(
              "invalid composed transition for `cfg`: %s (composed at %s)",
              e.getMessage(), composition.getLocation());
        }
      }
    }
    return result;
  }
}
