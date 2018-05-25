// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * A node in the build dependency graph, identified by a Label.
 */
@SkylarkModule(name = "target", doc = "", documented = false)
public interface TargetApi {

  /**
   * Returns the label of this target.  (e.g. "//foo:bar")
   */
  @SkylarkCallable(name = "label", documented = false)
  Label getLabel();

  /**
   * Returns the name of this rule (relative to its owning package).
   */
  @SkylarkCallable(name = "name", documented = false)
  String getName();
}
