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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provider for iOS Framework info.
 */
@Immutable
public final class IosFrameworkProvider implements TransitiveInfoProvider {

  private final String frameworkName;

  public IosFrameworkProvider(String frameworkName) {
    this.frameworkName = Preconditions.checkNotNull(frameworkName);
  }

  /**
   * Returns the name of the framework.
   */
  public String getFrameworkName() {
    return frameworkName;
  }
}
