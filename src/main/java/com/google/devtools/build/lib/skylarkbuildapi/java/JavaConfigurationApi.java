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

package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** A java compiler configuration. */
@SkylarkModule(
    name = "java",
    doc = "A java compiler configuration.",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT)
public interface JavaConfigurationApi extends StarlarkValue {

  @SkylarkCallable(
      name = "default_javac_flags",
      structField = true,
      doc = "The default flags for the Java compiler.")
  // TODO(bazel-team): this is the command-line passed options, we should remove from skylark
  // probably.
  ImmutableList<String> getDefaultJavacFlags();

  @SkylarkCallable(
      name = "strict_java_deps",
      structField = true,
      doc = "The value of the strict_java_deps flag.")
  String getStrictJavaDepsName();

  @SkylarkCallable(
      name = "plugins",
      structField = true,
      doc = "A list containing the labels provided with --plugins, if any.")
  ImmutableList<Label> getPlugins();
}
