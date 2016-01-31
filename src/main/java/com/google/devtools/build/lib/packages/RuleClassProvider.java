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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NativeAspectClass.NativeAspectFactory;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Mutability;

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
   * Returns a new Skylark Environment instance for rule creation.
   * Implementations need to be thread safe.
   * Be sure to close() the mutability before you return the results of said evaluation.
   * @param mutability the Mutability for the current evaluation context
   * @param eventHandler the EventHandler for warnings, errors, etc.
   * @param astFileContentHashCode the hash code identifying this environment.
   * @return an Environment, in which to evaluate load time skylark forms.
   */
  Environment createSkylarkRuleClassEnvironment(
      Mutability mutability,
      EventHandler eventHandler,
      @Nullable String astFileContentHashCode,
      @Nullable Map<String, Extension> importMap);

  /**
   * Returns a map from aspect names to aspect factory objects.
   */
  Map<String, Class<? extends NativeAspectFactory>> getAspectFactoryMap();

  /**
   * Returns the default content of the WORKSPACE file.
   *
   * <p>Used to provide external dependencies for built-in rules. Rules defined here can be
   * overwritten in the WORKSPACE file in the actual workspace.
   */
  String getDefaultWorkspaceFile();
  
  /**
   * Returns the path to the tools repository
   */
  String getToolsRepository();
}
