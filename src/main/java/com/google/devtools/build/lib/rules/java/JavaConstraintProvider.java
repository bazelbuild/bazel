// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;

/**
 * Provides information about the Java constraints (e.g. "android") that are
 * present on the target.
 */
@Immutable
public final class JavaConstraintProvider implements TransitiveInfoProvider {

  private final Label label;
  private final ImmutableList<String> javaConstraints;

  public JavaConstraintProvider(Label label, ImmutableList<String> javaConstraints) {
    this.label = label;
    this.javaConstraints = javaConstraints;
  }

  /**
   * Returns the label that is associated with this piece of information.
   *
   * <p>This is usually the label of the target that provides the information.
   */
  public Label getLabel() {
    return label;
  }

  /**
   * Returns all constraints set on the associated target.
   */
  public ImmutableList<String> getJavaConstraints() {
    return javaConstraints;
  }
}
