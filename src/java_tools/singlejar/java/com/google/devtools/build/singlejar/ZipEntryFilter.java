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

package com.google.devtools.build.singlejar;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Date;

/**
 * A custom filter for entries when combining multiple ZIP files (or even just
 * copying a single ZIP file).
 *
 * <p>Implementations of this interface must be thread-safe. The {@link
 * #accept} method may be called concurrently by multiple threads.
 */
public interface ZipEntryFilter {

  /**
   * Strategy for a custom merge operation. The current file and all additional
   * file are passed to the strategy object via {@link #merge}, which merges
   * the files. At the end of the ZIP combination, {@link #finish} is called,
   * which then writes the merged single entry of that name.
   *
   * <p>Implementations of this interface are not required to be thread-safe.
   * Thread-safety is achieved by creating multiple instances. Each instance
   * that is separately passed to {@link StrategyCallback#customMerge} is
   * guaranteed not to be called by two threads at the same time.
   */
  interface CustomMergeStrategy {

    /**
     * Merges another file into the current state. This method is called for
     * every file entry of the same name.
     */
    void merge(InputStream in, OutputStream out) throws IOException;

    /**
     * Outputs the merged result into the given output stream. This method is
     * only called once when no further file of the same name is available.
     */
    void finish(OutputStream out) throws IOException;

    /**
     * Called after {@link #finish} if no output was written to check if an empty file should be
     * written.  Returns {@code false} by default.
     * @return {@code true} to skip empty merge results, {@code false} to write them.
     */
    default boolean skipEmpty() {
      return false;
    }
  }

  /**
   * A callback interface for the {@link ZipEntryFilter#accept} method. Use
   * this interface to indicate the type of processing for the given file name.
   * For every file name, exactly one of the methods must be called once. A
   * second method call throws {@link IllegalStateException}.
   *
   * <p>There is no guarantee that the callback will perform the requested
   * operation at the time of the invocation. An implementation may choose to
   * defer the operation to an arbitrary later time.
   *
   * <p>IMPORTANT NOTE: Do not implement this interface. It will be modified to
   * support future extensions, and all implementations in this package will be
   * updated. If you violate this advice, your code will break.
   */
  interface StrategyCallback {

    /**
     * Skips the current entry and all entries with the same name.
     */
    void skip() throws IOException;

    /**
     * Copies the current entry and skips all further entries with the same
     * name. If {@code date} is non-null, then the timestamp of the entry is
     * overwritten with the given value.
     */
    void copy(Date date) throws IOException;

    /**
     * Renames and copies the current entry, and skips all further entries with
     * the same name. If {@code date} is non-null, then the timestamp of the entry
     * is overwritten with the given value.
     */
    void rename(String filename, Date date) throws IOException;

    /**
     * Merges this and all further entries with the same name with the given
     * {@link CustomMergeStrategy}. This method must never be called twice with
     * the same object. If {@code date} is non-null, then the timestamp of the
     * generated entry is set to the given value; otherwise, it is set to the
     * current time.
     */
    void customMerge(Date date, CustomMergeStrategy strategy) throws IOException;
  }

  /**
   * Determines the policy with which to handle the ZIP file entry with the
   * given name and calls the appropriate method on the callback interface
   * {@link StrategyCallback}. For every unique name in the set of all ZIP file
   * entries, this method is called exactly once and the result is used for all
   * entries of the same name. Except, if an entry is renamed, the original name
   * is not considered as having been encountered yet.
   *
   * <p>Implementations should use the filename to distinguish the desired
   * processing, call one method on the callback interface and return
   * immediately after that call.
   *
   * <p>There is no guarantee that the callback will perform the requested
   * operation at the time of the invocation. An implementation may choose to
   * defer the operation to an arbitrary later time.
   */
  void accept(String filename, StrategyCallback callback) throws IOException;
}
