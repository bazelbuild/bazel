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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.Root;
import java.util.Map;

/**
 * The collection of the supported build rules. Provides an StarlarkThread for Starlark rule
 * creation.
 */
public interface RuleClassProvider extends RuleDefinitionEnvironment {

  /** Label referencing the prelude file. */
  Label getPreludeLabel();

  /** Returns true if a package location is considered to be experimental. */
  boolean isPackageUnderExperimental(PackageIdentifier packageIdentifier);

  /** The runfiles prefix. */
  String getRunfilesPrefix();

  /**
   * Where the bundled builtins bzl files are located. These are the builtins files used if {@code
   * --experimental_builtins_bzl_path} is set to {@code %bundled%}. Note that this root lives in a
   * separate {@link InMemoryFileSystem}.
   *
   * <p>May be null in tests, in which case {@code --experimental_builtins_bzl_path} must point to
   * the builtins root to be used.
   */
  Root getBundledBuiltinsRoot();

  /**
   * The relative location of the builtins_bzl directory within a Bazel source tree.
   *
   * <p>May be null in tests, in which case --experimental_builtins_bzl_path may not be
   * "%workspace%".
   */
  String getBuiltinsBzlPackagePathInSource();

  /** Returns a map from rule names to rule class objects. */
  ImmutableMap<String, RuleClass> getRuleClassMap();

  /** Returns a map from aspect names to aspect factory objects. */
  Map<String, NativeAspectClass> getNativeAspectClassMap();

  /**
   * Returns the {@link BazelStarlarkEnvironment}, which is the final determiner of the BUILD and
   * .bzl environment (with and without builtins injection).
   */
  BazelStarlarkEnvironment getBazelStarlarkEnvironment();

  /** Retrieves an aspect from the aspect factory map using the key provided */
  NativeAspectClass getNativeAspectClass(String key);

  /**
   * Retrieves a {@link Map} from Starlark configuration fragment name to configuration fragment
   * class.
   */
  ImmutableMap<String, Class<?>> getConfigurationFragmentMap();
}
