// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.DependencyResolver.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.skylark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;

/** Exception class that signals an error during the evaluation of a dependency. */
public class DependencyEvaluationException extends Exception {
  public DependencyEvaluationException(InvalidConfigurationException cause) {
    super(cause);
  }

  public DependencyEvaluationException(ConfiguredValueCreationException cause) {
    super(cause);
  }

  public DependencyEvaluationException(InconsistentAspectOrderException cause) {
    super(cause);
  }

  public DependencyEvaluationException(TransitionException cause) {
    super(cause);
  }

  @Override
  public synchronized Exception getCause() {
    return (Exception) super.getCause();
  }
}
