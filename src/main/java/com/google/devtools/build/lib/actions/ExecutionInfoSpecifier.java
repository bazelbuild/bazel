// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import java.util.Map;

/**
 * An action that specifies requirements for its execution.
 */
public interface ExecutionInfoSpecifier {

  /**
   * Returns execution data for this action. This is used to signal hardware requirements to the
   * execution (ex. "requires-darwin"). This is a workaround to allow execution on platforms other
   * than that specified by the host configuration. The ability to choose separate platforms by
   * action can provide a performance advantage.
   *
   * <p>Restrictions are mapped to arbitrary values (typically "") so as to be consistent with
   * {@link com.google.devtools.build.lib.actions.Spawn#getExecutionInfo()}.
   */
  Map<String, String> getExecutionInfo();
}
