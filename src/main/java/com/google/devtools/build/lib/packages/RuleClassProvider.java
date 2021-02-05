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
import com.google.devtools.build.lib.analysis.RuleDefinitionContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.RuleClass.Builder.ThirdPartyLicenseExistencePolicy;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Map;
import net.starlark.java.eval.StarlarkThread;

/**
 * The collection of the supported build rules. Provides an StarlarkThread for Starlark rule
 * creation.
 */
public interface RuleClassProvider extends RuleDefinitionContext {

  /**
   * Label referencing the prelude file.
   */
  Label getPreludeLabel();

  /**
   * The default runfiles prefix (may be overwritten by the WORKSPACE file).
   */
  String getRunfilesPrefix();

  /**
   * Where the builtins bzl files are located (if not overridden by
   * --experimental_builtins_bzl_path). Note that this lives in a separate InMemoryFileSystem.
   *
   * <p>May be null in tests, in which case --experimental_builtins_bzl_path must point to a
   * builtins root.
   */
  Path getBuiltinsBzlRoot();

  /** The relative location of the builtins_bzl directory within a Bazel source tree. */
  String getBuiltinsBzlPackagePathInSource();

  /**
   * Returns a map from rule names to rule class objects.
   */
  Map<String, RuleClass> getRuleClassMap();

  /**
   * Stores a BazelStarlarkContext in the specified StarlarkThread about to initialize a .bzl file.
   *
   * <p>A .bzl file loaded by (or indirectly by) a BUILD file may differ semantically from the same
   * file loaded on behalf of a WORKSPACE file, because of the repository mapping and native module;
   * these differences much be accounted for by caching.
   *
   * @param thread StarlarkThread in which to store the context.
   * @param label the label of the .bzl file
   * @param repoMapping map of RepositoryNames to be remapped
   */
  void setStarlarkThreadContext(
      StarlarkThread thread,
      Label label,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping);

  /**
   * Returns all the predeclared top-level symbols (for .bzl files) that belong to native rule sets,
   * and hence are allowed to be overridden by builtins-injection.
   *
   * <p>For example, {@code CcInfo} is included, but {@code rule()} is not.
   *
   * @see StarlarkBuiltinsFunction
   */
  ImmutableMap<String, Object> getNativeRuleSpecificBindings();

  /**
   * Returns the Starlark builtins registered with this RuleClassProvider.
   *
   * <p>Does not account for builtins injection. Excludes universal bindings (e.g. True, len).
   *
   * <p>See {@link BazelStarlarkEnvironment#getUninjectedBuildBzlNativeBindings} for the canonical
   * determination of the bzl environment (before injection).
   */
  ImmutableMap<String, Object> getEnvironment();

  /**
   * Returns a map from aspect names to aspect factory objects.
   */
  Map<String, NativeAspectClass> getNativeAspectClassMap();

  /**
   * Returns the default content that should be added at the beginning of the WORKSPACE file.
   *
   * <p>Used to provide external dependencies for built-in rules. Rules defined here can be
   * overwritten in the WORKSPACE file in the actual workspace.
   */
  String getDefaultWorkspacePrefix();

  /**
   * Returns the default content that should be added at the end of the WORKSPACE file.
   *
   * <p>Used to load Starlark repository in the bazel_tools repository.
   */
  String getDefaultWorkspaceSuffix();

  /**
   * Retrieves an aspect from the aspect factory map using the key provided
   */
  NativeAspectClass getNativeAspectClass(String key);

  /**
   * Retrieves a {@link Map} from Starlark configuration fragment name to configuration fragment
   * class.
   */
  Map<String, Class<?>> getConfigurationFragmentMap();

  /** Returns the policy on checking that third-party rules have licenses. */
  ThirdPartyLicenseExistencePolicy getThirdPartyLicenseExistencePolicy();
}
