// Copyright 2022 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import java.util.HashMap;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkThread;

/** Class which can evaluate a label with repository remappings. */
public class LabelConverter {

  public static LabelConverter forThread(StarlarkThread thread) {
    BazelStarlarkContext bazelStarlarkContext = BazelStarlarkContext.from(thread);
    BazelModuleContext moduleContext =
        BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread));
    return new LabelConverter(
        moduleContext.label(),
        moduleContext.repoMapping(),
        bazelStarlarkContext.getConvertedLabelsInPackage());
  }

  private final Label parent;
  private final RepositoryMapping repositoryMapping;
  private final HashMap<String, Label> convertedLabelsInPackage;

  public LabelConverter(
      Label label,
      RepositoryMapping repositoryMapping,
      HashMap<String, Label> convertedLabelsInPackage) {
    this.parent = label;
    this.repositoryMapping = repositoryMapping;
    this.convertedLabelsInPackage = convertedLabelsInPackage;
  }

  Label getParent() {
    return parent;
  }

  /** Returns the Label corresponding to the input, using the current conversion context. */
  public Label convert(String input) throws LabelSyntaxException {
    // Optimization: First check the package-local map, avoiding Label validation, Label
    // construction, and global Interner lookup. This approach tends to be very profitable
    // overall, since it's common for the targets in a single package to have duplicate
    // label-strings across all their attribute values.
    Label converted = convertedLabelsInPackage.get(input);
    if (converted == null) {
      converted = parent.getRelativeWithRemapping(input, repositoryMapping);
      convertedLabelsInPackage.put(input, converted);
    }
    return converted;
  }

  RepositoryMapping getRepositoryMapping() {
    return repositoryMapping;
  }

  HashMap<String, Label> getConvertedLabelsInPackage() {
    return convertedLabelsInPackage;
  }

  @Override
  public String toString() {
    return parent.toString();
  }
}
