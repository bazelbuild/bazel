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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.actions.AbstractActionOwner;
import com.google.devtools.build.lib.actions.ActionOwner;

// TODO(bazel-team): move BUILD_INFO_ACTION_OWNER somewhere else and remove this class.
/**
 * Helper class for the CompatibleWriteBuildInfoAction, which holds the
 * methods for generating build information.
 * Abstracted away to allow non-action code to also generate build info under
 * --nobuild or --check_up_to_date.
 */
public abstract class BuildInfoHelper {
  /** ActionOwner for BuildInfoActions. */
  public static final ActionOwner BUILD_INFO_ACTION_OWNER =
      AbstractActionOwner.SYSTEM_ACTION_OWNER;
}
