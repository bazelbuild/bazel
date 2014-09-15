/*
 * Copyright 2013-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.facebook.buck.apple.xcode.xcodeproj;

/**
 * Build phase representing the linking step of a target. Contains references to libraries and
 * system frameworks. In library targets, it may alter the include path.
 */
public class PBXFrameworksBuildPhase extends PBXBuildPhase {
  @Override
  public String isa() {
    return "PBXFrameworksBuildPhase";
  }
}
