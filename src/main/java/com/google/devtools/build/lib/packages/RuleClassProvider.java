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
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkThread.Extension;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * The collection of the supported build rules. Provides an StarlarkThread for Skylark rule
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
   * Returns a map from rule names to rule class objects.
   */
  Map<String, RuleClass> getRuleClassMap();

  /**
   * Returns a new StarlarkThread for initialization of a .bzl file loaded on behalf of a BUILD or
   * WORKSPACE file. Implementations need to be thread safe. Be sure to close() the mutability
   * before you return the results of evaluation.
   *
   * <p>A .bzl file loaded by (or indirectly by) a BUILD file may differ semantically from the same
   * file loaded on behalf of a WORKSPACE file, because of the repository mapping and native module;
   * these differences much be accounted for by caching.
   *
   * @param label the label of the .bzl file
   * @param mutability the Mutability for the .bzl module globals
   * @param starlarkSemantics the semantics options that modify the interpreter
   * @param printHandler defines the behavior of Starlark print statements
   * @param astFileContentHashCode the hash code identifying this environment.
   * @param importMap map from import string to Extension
   * @param nativeModule the appropriate {@code native} module for this environment.
   * @param repoMapping map of RepositoryNames to be remapped
   * @return the StarlarkThread in which to initualize the .bzl module
   */
  StarlarkThread createRuleClassStarlarkThread(
      Label label,
      Mutability mutability,
      StarlarkSemantics starlarkSemantics,
      StarlarkThread.PrintHandler printHandler,
      @Nullable String astFileContentHashCode,
      @Nullable Map<String, Extension> importMap,
      ClassObject nativeModule,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping);

  /**
   * Returns the predeclared environment for a loading-phase thread. Includes "native", though its
   * value may be inappropriate for a WORKSPACE file. Includes the universal bindings (e.g. True,
   * len), though that will soon change.
   */
  // TODO(adonovan): update doc comment.
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
   * <p>Used to load skylark repository in the bazel_tools repository.
   */
  String getDefaultWorkspaceSuffix();

  /**
   * Retrieves an aspect from the aspect factory map using the key provided
   */
  NativeAspectClass getNativeAspectClass(String key);

  /**
   * Retrieves a {@link Map} from skylark configuration fragment name to configuration fragment
   * class.
   */
  Map<String, Class<?>> getConfigurationFragmentMap();

  /** Returns the policy on checking that third-party rules have licenses. */
  ThirdPartyLicenseExistencePolicy getThirdPartyLicenseExistencePolicy();
}
