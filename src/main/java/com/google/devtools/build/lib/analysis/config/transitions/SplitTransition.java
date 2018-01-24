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

package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import java.util.List;

/**
 * A configuration split transition; this should be used to transition to multiple configurations
 * simultaneously. Note that the corresponding rule implementations must have special support to
 * handle this.
 */
@ThreadSafety.Immutable
@FunctionalInterface
public interface SplitTransition extends Transition {
  /**
   * Return the list of {@code BuildOptions} after splitting; empty if not applicable.
   */
  List<BuildOptions> split(BuildOptions buildOptions);
}
