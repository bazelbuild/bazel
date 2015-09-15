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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.syntax.Label;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * A rule visibility that simply says yes or no. It corresponds to public,
 * legacy_public and private visibilities.
 */
@Immutable @ThreadSafe
public class ConstantRuleVisibility implements RuleVisibility, Serializable {
  static final Label LEGACY_PUBLIC_LABEL;  // same as "public"; used for automated depot cleanup
  private static final Label PUBLIC_LABEL;
  private static final Label PRIVATE_LABEL;

  public static final ConstantRuleVisibility PUBLIC =
      new ConstantRuleVisibility(true);

  public static final ConstantRuleVisibility PRIVATE =
      new ConstantRuleVisibility(false);

  static {
    try {
      PUBLIC_LABEL = Label.parseAbsolute("//visibility:public");
      LEGACY_PUBLIC_LABEL = Label.parseAbsolute("//visibility:legacy_public");
      PRIVATE_LABEL = Label.parseAbsolute("//visibility:private");
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException();
    }
  }

  private final boolean result;

  public ConstantRuleVisibility(boolean result) {
    this.result = result;
  }

  public boolean isPubliclyVisible() {
    return result;
  }

  @Override
  public List<Label> getDependencyLabels() {
    return Collections.emptyList();
  }

  @Override
  public List<Label> getDeclaredLabels() {
    return ImmutableList.of(result ? PUBLIC_LABEL : PRIVATE_LABEL);
  }

  /**
   * Tries to parse a list of labels into a {@link ConstantRuleVisibility}.
   *
   * @param labels the list of labels to parse
   * @return The resulting visibility object, or null if the list of labels
   * could not be parsed.
   */
  public static ConstantRuleVisibility tryParse(List<Label> labels) {
    if (labels.size() != 1) {
      return null;
    }
    return tryParse(labels.get(0));
  }

  public static ConstantRuleVisibility tryParse(Label label) {
    if (PUBLIC_LABEL.equals(label) || LEGACY_PUBLIC_LABEL.equals(label)) {
      return PUBLIC;
    } else if (PRIVATE_LABEL.equals(label)) {
      return PRIVATE;
    } else {
      return null;
    }
  }
}
