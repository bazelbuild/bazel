// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.packages.Attribute;

/**
 * Interface for a configuration transition using dynamic configurations.
 *
 * <p>The concept is simple: given the input configuration's build options, the
 * transition does whatever it wants to them and returns the modified result.
 *
 * <p>Implementations must be stateless: the transformation logic cannot use
 * any information from any data besides the input build options.
 *
 * <p>For performance reasons, the input options are passed in as a <i>reference</i>,
 * not a <i>copy</i>. Transition implementations should <i>always</i> treat these
 * options as immutable, and call
 * {@link com.google.devtools.build.lib.analysis.config.BuildOptions#clone}
 * before applying any mutations. Unfortunately,
 * {@link com.google.devtools.build.lib.analysis.config.BuildOptions} does not currently
 * enforce immutability, so care must be taken not to modify the wrong instance.
 */
public interface PatchTransition extends Attribute.Transition {

  /**
   * Applies the transition.
   *
   * @param options the options representing the input configuration to this transition. DO NOT
   *     MODIFY THIS VARIABLE WITHOUT CLONING IT FIRST.
   * @return the options representing the desired post-transition configuration
   */
  BuildOptions apply(BuildOptions options);
}
