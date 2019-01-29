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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * The C++ configuration fragment.
 */
@SkylarkModule(
  name = "cpp",
  doc = "A configuration fragment for C++.",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
public interface CppConfigurationApi <InvalidConfigurationExceptionT extends Exception> {

  @SkylarkCallable(
      name = "copts",
      structField = true,
      doc = "Returns flags passed to Bazel by --copt option.")
  ImmutableList<String> getCopts() throws EvalException;

  @SkylarkCallable(
      name = "cxxopts",
      structField = true,
      doc = "Returns flags passed to Bazel by --cxxopt option.")
  ImmutableList<String> getCxxopts() throws EvalException;

  @SkylarkCallable(
      name = "conlyopts",
      structField = true,
      doc = "Returns flags passed to Bazel by --conlyopt option.")
  ImmutableList<String> getConlyopts() throws EvalException;

  @SkylarkCallable(
      name = "linkopts",
      structField = true,
      doc = "Returns flags passed to Bazel by --linkopt option.")
  ImmutableList<String> getLinkopts() throws EvalException;
}
