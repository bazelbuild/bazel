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

package com.google.devtools.build.lib.starlarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** A java compiler configuration. */
@StarlarkBuiltin(
    name = "java",
    doc = "A java compiler configuration.",
    category = DocCategory.CONFIGURATION_FRAGMENT)
public interface JavaConfigurationApi extends StarlarkValue {

  @StarlarkMethod(
      name = "default_javac_flags",
      structField = true,
      doc = "The default flags for the Java compiler.")
  // TODO(bazel-team): this is the command-line passed options, we should remove from Starlark
  // probably.
  ImmutableList<String> getDefaultJavacFlags();

  @StarlarkMethod(
      name = "strict_java_deps",
      structField = true,
      doc = "The value of the strict_java_deps flag.")
  String getStrictJavaDepsName();

  @StarlarkMethod(
      name = "default_jvm_opts",
      structField = true,
      doc = "Additional options to pass to the Java VM for each java_binary target")
  ImmutableList<String> getDefaultJvmFlags();

  @StarlarkMethod(
      name = "one_version_enforcement_level",
      structField = true,
      doc = "The value of the --experimental_one_version_enforcement flag.")
  String starlarkOneVersionEnforcementLevel();

  @StarlarkMethod(
      name = "plugins",
      structField = true,
      doc = "A list containing the labels provided with --plugins, if any.")
  ImmutableList<Label> getPlugins();
}
