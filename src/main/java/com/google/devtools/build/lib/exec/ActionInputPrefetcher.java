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
package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.ActionInput;

/** Prefetches files to local disk. */
public interface ActionInputPrefetcher {

  /**
   * Initiates best-effort prefetching of the given input. This should not block.
   *
   * <p>If the path is not under this prefetcher's control, the call should be a no-op.
   */
  void prefetchFile(ActionInput input);
}
