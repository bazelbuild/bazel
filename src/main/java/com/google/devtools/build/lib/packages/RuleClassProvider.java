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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * The collection of the supported build rules. Provides an Environment for Skylark rule creation.
 */
public interface RuleClassProvider {

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
   * Returns a new Skylark Environment instance for rule creation. Implementations need to be thread
   * safe. Be sure to close() the mutability before you return the results of said evaluation.
   *
   * @param label the location of the rule.
   * @param mutability the Mutability for the current evaluation context
   * @param starlarkSemantics the semantics options that modify the interpreter
   * @param eventHandler the EventHandler for warnings, errors, etc.
   * @param astFileContentHashCode the hash code identifying this environment.
   * @param importMap map from import string to Extension
   * @param repoMapping map of RepositoryNames to be remapped
   * @return an Environment, in which to evaluate load time skylark forms.
   */
  Environment createSkylarkRuleClassEnvironment(
      Label label,
      Mutability mutability,
      StarlarkSemantics starlarkSemantics,
      EventHandler eventHandler,
      @Nullable String astFileContentHashCode,
      @Nullable Map<String, Extension> importMap,
      ImmutableMap<RepositoryName, RepositoryName> repoMapping);

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
   * Returns the path to the tools repository
   */
  String getToolsRepository();

  /**
   * Retrieves an aspect from the aspect factory map using the key provided
   */
  NativeAspectClass getNativeAspectClass(String key);

  /**
   * Retrieves a {@link Map} from skylark configuration fragment name to configuration fragment
   * class.
   */
  Map<String, Class<?>> getConfigurationFragmentMap();
}
