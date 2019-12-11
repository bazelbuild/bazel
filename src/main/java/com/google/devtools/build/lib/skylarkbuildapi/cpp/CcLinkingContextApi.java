// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Wrapper for every C++ linking provider. */
@SkylarkModule(
    name = "LinkingContext",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "Immutable store of information needed for C++ linking that is aggregated across "
            + "dependencies.")
public interface CcLinkingContextApi<FileT extends FileApi> extends StarlarkValue {
  @SkylarkCallable(
      name = "user_link_flags",
      doc = "Returns the list of user link flags passed as strings.",
      structField = true)
  Sequence<String> getSkylarkUserLinkFlags();

  @SkylarkCallable(
      name = "libraries_to_link",
      doc =
          "Returns the depset of <code>LibraryToLink</code>. May return a list but this is"
              + "deprecated. See #8118.",
      structField = true,
      useStarlarkSemantics = true)
  Object getSkylarkLibrariesToLink(StarlarkSemantics semantics);

  @SkylarkCallable(
      name = "additional_inputs",
      doc = "Returns the depset of additional inputs, e.g.: linker scripts.",
      structField = true)
  Depset getSkylarkNonCodeInputs();

  @SkylarkCallable(
      name = "linker_inputs",
      doc = "Returns the depset of linker inputs.",
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_CC_SHARED_LIBRARY,
      structField = true)
  Depset getSkylarkLinkerInputs();
}
