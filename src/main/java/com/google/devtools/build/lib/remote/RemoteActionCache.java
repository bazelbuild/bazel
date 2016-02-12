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

package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.Collection;

/**
 * A cache for storing artifacts (input and output) as well as the output of running an action.
 */
@ThreadCompatible
interface RemoteActionCache {
  /**
   * Put the file in cache if it is not already in it. No-op if the file is already stored in
   * cache.
   *
   * @return The key for fetching the file from cache.
   */
  String putFileIfNotExist(Path file) throws IOException;

  /**
   * Same as {@link putFileIfNotExist(Path)} but this methods takes an ActionInput.
   *
   * @return The key for fetching the file from cache.
   */
  String putFileIfNotExist(ActionInputFileCache cache, ActionInput file) throws IOException;

  /**
   * Write the file in cache identified by key to the file system. The key must uniquely identify
   * the content of the file. Throws CacheNotFoundException if the file is not found in cache.
   */
  void writeFile(String key, Path dest, boolean executable)
      throws IOException, CacheNotFoundException;

  /**
   * Write the action output files identified by the key to the file system. The key must uniquely
   * identify the action and the content of action inputs.
   *
   * @throws CacheNotFoundException if action output is not found in cache.
   */
  void writeActionOutput(String key, Path execRoot)
      throws IOException, CacheNotFoundException;

  /**
   * Update the cache with the action outputs for the specified key.
   */
  void putActionOutput(String key, Collection<? extends ActionInput> outputs)
      throws IOException;

  /**
   * Update the cache with the files for the specified key.
   */
  void putActionOutput(String key, Path execRoot, Collection<Path> files) throws IOException;
}
