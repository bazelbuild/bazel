// Copyright 2017 The Bazel Authors. All rights reserved.
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

/**
 * Retrieves {@link TransitiveOptionDetails} from {@link BuildConfiguration} instances.
 *
 * <p>This class's existence allows for the use of Blaze visibility to limit access to option data
 * to only the configuration-specific rules which need to access or manipulate the configuration in
 * such a meta way - in most cases, there should be no need to use this class. Instead, access
 * desired configuration fragments via {@link BuildConfiguration#getFragment(Class)}.
 */
public class BuildConfigurationOptionDetails {

  /** Utility class - no need to instantiate. */
  private BuildConfigurationOptionDetails() {}

  public static TransitiveOptionDetails get(BuildConfiguration configuration) {
    return configuration.getTransitiveOptionDetails();
  }
}
