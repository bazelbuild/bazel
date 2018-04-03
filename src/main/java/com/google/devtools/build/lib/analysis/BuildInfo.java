// Copyright 2015 The Bazel Authors. All rights reserved.
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

/**
 * A class to hold and query the build information and support methods to read
 * it from disk.
 */
public class BuildInfo {
  /**
   * Named constants for the BuildInfo keys.
   */
  public static final String BUILD_EMBED_LABEL = "BUILD_EMBED_LABEL";

  /**
   * The name of the user that performs the build.
   */
  public static final String BUILD_USER = "BUILD_USER";

  /**
   * The host where the build happens
   */
  public static final String BUILD_HOST = "BUILD_HOST";

  /**
   * Build time as seconds since epoch
   */
  public static final String BUILD_TIMESTAMP = "BUILD_TIMESTAMP";

  /**
   * The revision of source tree reported by source control system
   */
  public static final String BUILD_SCM_REVISION = "BUILD_SCM_REVISION";

  /**
   * The status of source tree reported by source control system
   */
  public static final String BUILD_SCM_STATUS = "BUILD_SCM_STATUS";
}
