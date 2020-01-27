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

package com.google.devtools.build.lib.actions;

/**
 * This class provide a fallback of the local host's resource capacity.
 */
public class LocalHostResourceFallback {

  /* If /proc/* information is not available, guess based on what the JVM thinks.  Anecdotally,
   * the JVM picks 0.22 the total available memory as maxMemory (tested on a standard Mac), so
   * multiply by 3, and divide by 2^20 because we want megabytes.
   */
  private static final ResourceSet DEFAULT_RESOURCES =
      ResourceSet.create(
          3.0 * (Runtime.getRuntime().maxMemory() >> 20),
          Runtime.getRuntime().availableProcessors(),
          Integer.MAX_VALUE);

  public static ResourceSet getLocalHostResources() {
    return DEFAULT_RESOURCES;
  }
}
