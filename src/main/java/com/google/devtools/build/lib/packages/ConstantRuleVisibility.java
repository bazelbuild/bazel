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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * A rule visibility that simply says yes or no. It corresponds to public and private visibilities.
 */
@Immutable
@ThreadSafe
public class ConstantRuleVisibility implements RuleVisibility, Serializable {
  @AutoCodec @AutoCodec.VisibleForSerialization static final Label PUBLIC_LABEL;
  @AutoCodec @AutoCodec.VisibleForSerialization static final Label PRIVATE_LABEL;

  @AutoCodec public static final ConstantRuleVisibility PUBLIC = new ConstantRuleVisibility(true);

  @AutoCodec public static final ConstantRuleVisibility PRIVATE = new ConstantRuleVisibility(false);

  static {
    try {
      PUBLIC_LABEL = Label.parseAbsolute("//visibility:public", ImmutableMap.of());
      PRIVATE_LABEL = Label.parseAbsolute("//visibility:private", ImmutableMap.of());
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
  public static ConstantRuleVisibility tryParse(List<Label> labels) throws EvalException {
    if (labels.size() == 1) {
      return tryParse(labels.get(0));
    }
    ConstantRuleVisibility visibility;
    for (Label label : labels) {
      visibility = tryParse(label);
      if (visibility != null) {
        throw Starlark.errorf(
            "Public or private visibility labels (e.g. //visibility:public or"
                + " //visibility:private) cannot be used in combination with other labels");
      }
    }
    return null;
  }

  public static ConstantRuleVisibility tryParse(Label label) {
    if (PUBLIC_LABEL.equals(label)) {
      return PUBLIC;
    } else if (PRIVATE_LABEL.equals(label)) {
      return PRIVATE;
    } else {
      return null;
    }
  }
}
