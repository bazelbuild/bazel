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

package com.google.devtools.build.lib.view.filegroup;

import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

/**
 * A provider implemented by {@link FilegroupConfiguredTarget} so that dependent targets can query
 * its {@code path} attribute.
 */
public interface FilegroupPathProvider extends TransitiveInfoProvider {
  /**
   * Returns the value of the {@code path} attribute or the empty fragment if it is not present.
   */
  PathFragment getFilegroupPath();
}
