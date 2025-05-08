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
package com.google.devtools.build.lib.packages.producers;

import com.google.auto.value.AutoOneOf;
import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionException;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;

/** Tagged union of possible errors that can be accepted by {@link GlobComputationProducer}. */
@AutoOneOf(GlobError.Kind.class)
public abstract class GlobError {
  /** Tags the error type. */
  public enum Kind {
    INCONSISTENT_FILESYSTEM,
    FILE_SYMLINK_INFINITE_EXPANSION
  }

  public abstract Kind kind();

  public abstract InconsistentFilesystemException inconsistentFilesystem();

  public abstract FileSymlinkInfiniteExpansionException fileSymlinkInfiniteExpansion();

  public static GlobError of(InconsistentFilesystemException e) {
    return AutoOneOf_GlobError.inconsistentFilesystem(e);
  }

  public static GlobError of(FileSymlinkInfiniteExpansionException e) {
    return AutoOneOf_GlobError.fileSymlinkInfiniteExpansion(e);
  }
}
