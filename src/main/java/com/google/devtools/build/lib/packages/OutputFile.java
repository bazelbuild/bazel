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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.syntax.Location;

/**
 * A generated file that is the output of a rule.
 */
public final class OutputFile extends FileTarget {

  private final Rule generatingRule;

  /**
   * Constructs an OutputFile with the given label, which must be in the generating rule's package.
   */
  OutputFile(Label label, Rule generatingRule) {
    super(generatingRule.getPackage(), label);
    this.generatingRule = generatingRule;
  }

  @Override
  public RuleVisibility getVisibility() {
    return generatingRule.getVisibility();
  }

  @Override
  public boolean isConfigurable() {
    return true;
  }

  /**
   * Returns the rule which generates this output file.
   */
  public Rule getGeneratingRule() {
    return generatingRule;
  }

  @Override
  public Package getPackage() {
    return generatingRule.getPackage();
  }

  /**
   * A kind of output file.
   *
   * <p>The FILESET kind is only supported for a non-open-sourced {@code fileset} rule.
   */
  public enum Kind {
    FILE,
    FILESET
  }

  /**
   * Returns the kind of this output file.
   */
  public Kind getKind() {
    return generatingRule.getRuleClassObject().getOutputFileKind();
  }

  @Override
  public String getTargetKind() {
    return targetKind();
  }

  @Override
  public Rule getAssociatedRule() {
    return generatingRule;
  }

  @Override
  public Location getLocation() {
    return generatingRule.getLocation();
  }

  /** Returns the target kind for all output files. */
  public static String targetKind() {
    return "generated file";
  }
}
