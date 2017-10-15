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
package com.google.devtools.build.lib.rules.fileset;

import com.google.devtools.build.lib.actions.ActionContext;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Action context for fileset collection actions.
 */
public interface FilesetActionContext extends ActionContext {

  /**
   * Returns a thread pool for fileset symlink tree creation.
   */
  ThreadPoolExecutor getFilesetPool();

  /**
   * Returns the name of the workspace the build is run in.
   */
  String getWorkspaceName();
}
