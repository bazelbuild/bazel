// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;

/** A MODULE.bazel file's content and location. */
@AutoValue
public abstract class ModuleFile {
  /** The raw content of the module file. */
  @SuppressWarnings("mutable")
  public abstract byte[] getContent();

  /** The user-facing location of the module file, e.g. a file system path or URL. */
  public abstract String getLocation();

  public static ModuleFile create(byte[] content, String location) {
    return new AutoValue_ModuleFile(content, location);
  }
}
