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
package com.google.devtools.build.lib.syntax;

import javax.annotation.Nullable;

/**
 * Interface for retrieving the name of a {@link
 * com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment} based on its
 * class.
 * 
 * <p>Classes implementing this specific interface are required when doing look up operations and
 * comparisons of configuration fragments.
 */
public interface FragmentClassNameResolver {
  /**
   * Returns the name of the configuration fragment specified by the given class or null, if this is
   * not possible.
   */
  @Nullable
  String resolveName(Class<?> clazz);
}
