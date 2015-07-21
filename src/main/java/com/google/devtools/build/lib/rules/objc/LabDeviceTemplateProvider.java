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
package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provides template which can be run by the test runner script of {@code experimental_ios_test}
 * targets for lab devices.
 *
 * <p> The template contained in a {@code ios_lab_device} target.
 */
@Immutable
public final class LabDeviceTemplateProvider implements TransitiveInfoProvider {
  private final Artifact template;

  public LabDeviceTemplateProvider(Artifact template) {
    this.template = Preconditions.checkNotNull(template);
  }

  /**
   * Returns the template for lab devices.
   *
   * <p>The template contains the following substitution variables:
   * <ul>
   * <li> the %launcher and %launcher_arg (whose values must separately be provided through
   *      an {@link IosTestSubstitutionProvider})
   * <li> %test_app_ipa, %test_app_name, %xctest_app_ipa, %xctest_app_name, %ios_device_arg
   *      (whose values are expected to be provided by the substituting rule)
   * </ul>
   */
  public Artifact getLabDeviceTemplate() {
    return template;
  }
}
