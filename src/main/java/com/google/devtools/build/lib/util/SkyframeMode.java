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
package com.google.devtools.build.lib.util;

/**
 * How much of Skyframe should be used for a build.
 */
public enum SkyframeMode {
  LOADING_AND_ANALYSIS (1),
  FULL (2),
  EXPERIMENTAL_FULL (3); // TODO(bazel-team): Remove after a few Blaze releases.

  private final int order;

  private SkyframeMode(int order) {
    this.order = order;
  }

  public boolean atLeast(SkyframeMode lesser) {
    return this.order >= lesser.order;
  }

  public boolean atMost(SkyframeMode lesser) {
    return this.order <= lesser.order;
  }
}
