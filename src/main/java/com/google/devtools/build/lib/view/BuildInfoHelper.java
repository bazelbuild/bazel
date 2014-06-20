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

package com.google.devtools.build.lib.view;

import com.google.devtools.build.lib.actions.AbstractActionOwner;

/**
 * Helper class for the CompatibleWriteBuildInfoAction, which holds the
 * methods for generating build information.
 * Abstracted away to allow non-action code to also generate build info under
 * --nobuild or --check_up_to_date.
 */
public abstract class BuildInfoHelper {
  /** ActionOwner for BuildInfoActions. */
  public static final AbstractActionOwner BUILD_INFO_ACTION_OWNER = new AbstractActionOwner() {
    @Override
    public final String getConfigurationName() {
      return "system";
    }

    @Override
    public String getConfigurationMnemonic() {
      return "system";
    }

    @Override
    public final String getConfigurationShortCacheKey() {
      return "system";
    }
  };
}
