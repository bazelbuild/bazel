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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Either libraries, flags or other files that may be passed to the linker as inputs. */
@SkylarkModule(
    name = "LinkerInput",
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Either libraries, flags or other files that may be passed to the linker as inputs.")
public interface LinkerInputApi<
        LibraryToLinkT extends LibraryToLinkApi<FileT>, FileT extends FileApi>
    extends StarlarkValue {
  @SkylarkCallable(
      name = "owner",
      doc = "Returns the owner of this LinkerInput.",
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_CC_SHARED_LIBRARY,
      useLocation = true,
      structField = true)
  Label getSkylarkOwner(Location location) throws EvalException;

  @SkylarkCallable(
      name = "user_link_flags",
      doc = "Returns the list of user link flags passed as strings.",
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_CC_SHARED_LIBRARY,
      structField = true)
  Sequence<String> getSkylarkUserLinkFlags();

  @SkylarkCallable(
      name = "libraries",
      doc =
          "Returns the depset of <code>LibraryToLink</code>. May return a list but this is "
              + "deprecated. See #8118.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_CC_SHARED_LIBRARY,
      useStarlarkSemantics = true)
  Sequence<LibraryToLinkT> getSkylarkLibrariesToLink(StarlarkSemantics semantics);

  @SkylarkCallable(
      name = "additional_inputs",
      doc = "Returns the depset of additional inputs, e.g.: linker scripts.",
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_CC_SHARED_LIBRARY,
      structField = true)
  Sequence<FileT> getSkylarkNonCodeInputs();
}
