// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.vfs.Dirent;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Interface for both iterating over the entries in a directory and getting the entry, if any, for a
 * given basename.
 */
public interface Dirents extends Collection<Dirent> {

  @Nullable
  Dirent maybeGetDirent(String baseName);
}
