// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.cmdline.Label;

/** Observes transitions for Cquery. */
public interface TransitionCollector {
  public static final TransitionCollector NULL_TRANSITION_COLLECTOR = (k, l, transition) -> {};

  /**
   * The implementation of dependency resolution calls this as transitions are created.
   *
   * <p>This does not need to be called for no or null transitions.
   */
  void acceptTransition(DependencyKind kind, Label label, ConfigurationTransition transition);
}
