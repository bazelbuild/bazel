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

package com.google.devtools.build.lib.packages;

/**
 * A feature of the build that can be switched on and off on a per-package
 * basis.
 *
 * <p>This interface is only for marking targets as being affected by a feature;
 * their implementation can be anywhere.
 *
 * Implementations of this interface must be immutable.
 */
public interface DefaultSetting {
  String getSettingName();

  /**
   * Returns if the default setting in question affects the specific target.
   */
  boolean appliesTo(Target target);
}
