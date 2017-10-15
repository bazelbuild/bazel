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
package com.google.devtools.build.android.junctions;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import javax.annotation.Nullable;

/**
 * Interface to create junctions (directory symlinks).
 *
 * <p>Junctions are directory symlinks on NTFS filesystems. They are useful on Windows, because
 * creating them doesn't require any privileges, as opposed to the creation of file symlinks which
 * does.
 *
 * <p>On Windows, Bazel and the Android BusyBox uses junctions to work around path length
 * limitations of the Windows Shell and of tools like aapt.exe and the PNG cruncher. The limit is
 * 260 characters for all paths. The filesystem supports longer paths than that, but the tools
 * usually don't. To work around that limitation, we create junctions that have short paths but
 * point to long paths (this is allowed).
 *
 * <p>On Linux/MacOS the junction creator may have a no-op implementation.
 */
public interface JunctionCreator extends Closeable {
  /**
   * Returns an equivalent path to `target`, which may or may not be the same as `target`.
   *
   * <p>Depending on the implementation, this method may return `target` itself, or may create a
   * junction that points to `target` (if `target` is a directory) or the parent of it (if `target`
   * is a file).
   */
  @Nullable
  public abstract Path create(@Nullable Path target) throws IOException;
}
