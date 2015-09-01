// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.ValidationEnvironment;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Map;

/**
 * The collection of the supported build rules. Provides an Environment for Skylark rule creation.
 */
public interface RuleClassProvider {

  /**
   * Workspace relative path to the prelude file.
   */
  PathFragment getPreludePath();

  /**
   * Returns a map from rule names to rule class objects.
   */
  Map<String, RuleClass> getRuleClassMap();

  /**
   * Returns a map from aspect names to aspect factory objects.
   */
  Map<String, Class<? extends AspectFactory<?, ?, ?>>> getAspectFactoryMap();

  /**
   * Returns a new Skylark Environment instance for rule creation. Implementations need to be
   * thread safe.
   */
  SkylarkEnvironment createSkylarkRuleClassEnvironment(
      EventHandler eventHandler, String astFileContentHashCode);

  /**
   * Returns a validation environment for static analysis of skylark files.
   * The environment has to contain all built-in functions and objects.
   */
  ValidationEnvironment getSkylarkValidationEnvironment();

  /**
   * Returns the default content of the WORKSPACE file.
   *
   * <p>Used to provide external dependencies for built-in rules. Rules defined here can be
   * overwritten in the WORKSPACE file in the actual workspace.
   */
  String getDefaultWorkspaceFile();
}
