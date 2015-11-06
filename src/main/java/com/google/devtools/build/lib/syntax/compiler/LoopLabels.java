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
package com.google.devtools.build.lib.syntax.compiler;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.syntax.FlowStatement.Kind;

import org.objectweb.asm.Label;

/**
 * Container for passing around current loop labels during compilation.
 */
public class LoopLabels {
  public final Label continueLabel;
  public final Label breakLabel;

  private LoopLabels(Label continueLabel, Label breakLabel) {
    this.continueLabel = continueLabel;
    this.breakLabel = breakLabel;
  }

  /**
   * Only use Optional-wrapped instances of this class.
   */
  public static Optional<LoopLabels> of(Label continueLabel, Label breakLabel) {
    return Optional.of(new LoopLabels(continueLabel, breakLabel));
  }

  public static final Optional<LoopLabels> ABSENT = Optional.absent();

  /**
   * Get the label for a certain kind of flow statement.
   */
  public Label labelFor(Kind kind) {
    switch (kind) {
      case BREAK:
        return breakLabel;
      case CONTINUE:
        return continueLabel;
      default:
        throw new Error("missing flow kind: " + kind);
    }
  }
}
