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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** A representation of the concept "this builds these files". */
@StarlarkBuiltin(
    name = "file_provider",
    doc = "An interface for rules that provide files.",
    category = StarlarkDocumentationCategory.PROVIDER)
public interface FileProviderApi extends StarlarkValue {

  /**
   * Returns the set of files that are the "output" of this rule.
   *
   * <p>The term "output" is somewhat hazily defined; it is vaguely the set of files that are passed
   * on to dependent rules that list the rule in their {@code srcs} attribute and the set of files
   * that are built when a rule is mentioned on the command line. It does <b>not</b> include the
   * runfiles.
   *
   * <p>Note that the above definition is somewhat imprecise; in particular, when a rule is
   * mentioned on the command line, some other files are also built and dependent rules are free to
   * filter this set of files e.g. based on their extension.
   *
   * <p>Also, some rules may generate files that are not listed here by way of defining other
   * implicit targets, for example, deploy jars.
   */
  @StarlarkMethod(name = "files_to_build", documented = false, structField = true)
  Depset /*<? extends FileApi>*/ getFilesToBuildForStarlark();
}
