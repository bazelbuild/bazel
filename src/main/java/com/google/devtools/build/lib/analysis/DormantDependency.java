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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;

/**
 * Not an actual dependency, but the possibility of one.
 *
 * <p>Dormant attributes result in an instance of this object for each possible dependency edge. It
 * can then be passed up the dependency graph and turned into an actual dependency ("materialized")
 * by rules in the reverse transitive closure.
 */
public record DormantDependency(Label label) implements StarlarkValue {
  public static final String NAME = "dormant_dependency";
  public static final String ALLOWLIST_ATTRIBUTE_NAME = "$allowlist_dormant_dependency";
  public static final String ALLOWLIST_LABEL_STR =
      "//tools/allowlists/dormant_dependency_allowlist";
  public static final Label ALLOWLIST_LABEL = Label.parseCanonicalUnchecked(ALLOWLIST_LABEL_STR);

  @Override
  public void repr(Printer printer) {
    printer.append("<dormant dependency label='");
    printer.append(label.toString());
    printer.append("'>");
  }

  @StarlarkMethod(name = "label", doc = "TBD")
  public Label getLabel() {
    return label;
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public String toString() {
    return "<dormant dependency " + label.toString() + ">";
  }
}
