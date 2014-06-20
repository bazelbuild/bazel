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

package com.google.devtools.build.lib.vfs;


/**
 * An interface supporting extraction of workspace information from {@link PathFragment}s.
 */
public interface WorkspaceExtractor {
  /**
   * Extracts information about the user's workspace from the given path.
   */
  public AbstractWorkspace extractWorkspace(PathFragment path);

  /**
   * Removes workspace information from the given PathFragment.
   */
  public PathFragment stripWorkspace(PathFragment path);
}
