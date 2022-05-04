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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import java.util.HashMap;
import java.util.Map;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkThread;

/** Class which can evaluate a label with repository remappings. */
public class LabelConverter {

  public static LabelConverter forThread(StarlarkThread thread) {
    BazelModuleContext moduleContext =
        BazelModuleContext.of(Module.ofInnermostEnclosingStarlarkFunction(thread));
    BazelStarlarkContext bazelStarlarkContext = BazelStarlarkContext.from(thread);
    return new LabelConverter(
        moduleContext.label(),
        moduleContext.repoMapping(),
        bazelStarlarkContext.getConvertedLabelsInPackage());
  }

  public static LabelConverter forModuleContext(BazelModuleContext moduleContext) {
    return new LabelConverter(moduleContext.label(), moduleContext.repoMapping());
  }

  private final Label base;
  private final RepositoryMapping repositoryMapping;
  private final Map<String, Label> labelCache;

  public LabelConverter(Label base, RepositoryMapping repositoryMapping) {
    this(
        base,
        repositoryMapping,
        // Only cache labels seen by this converter.
        new HashMap<>());
  }

  public LabelConverter(
      Label base, RepositoryMapping repositoryMapping, Map<String, Label> labelCache) {
    this.base = base;
    this.repositoryMapping = repositoryMapping;
    this.labelCache = labelCache;
  }

  /** Returns the base label that relative labels will be resolved against. */
  Label getBase() {
    return base;
  }

  /** Returns the Label corresponding to the input, using the current conversion context. */
  public Label convert(String input) throws LabelSyntaxException {
    // Optimization: First check the package-local map, avoiding Label validation, Label
    // construction, and global Interner lookup. This approach tends to be very profitable
    // overall, since it's common for the targets in a single package to have duplicate
    // label-strings across all their attribute values.
    Label converted = labelCache.get(input);
    if (converted == null) {
      converted = base.getRelativeWithRemapping(input, repositoryMapping);
      labelCache.put(input, converted);
    }
    return converted;
  }

  @VisibleForTesting
  Map<String, Label> getLabelCache() {
    return labelCache;
  }

  @Override
  public String toString() {
    return base.toString();
  }
}
