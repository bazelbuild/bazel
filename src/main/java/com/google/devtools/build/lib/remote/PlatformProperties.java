// Copyright 2026 The Bazel Authors. All rights reserved.
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

/**
 * Strings used to communicate properties through the Remote Execution API.
 *
 * <p>While these can interact with {@link
 * com.google.devtools.build.lib.actions.ExecutionRequirements}, platform properties specifically
 * refer Property entries inserted into build.bazel.remote.execution.v2.Platform remote execution
 * messages. This allows Bazel to communicate per-action requirements to a remote execution service.
 *
 * <p>Prefer formally introducing new platform properties to the Remote Execution API before adding
 * them here.
 */
public class PlatformProperties {
  /**
   * (nonstandard) A unique identifier used to group identical build actions.
   *
   * <p>The value of this string indicates which persistent worker should handle incoming work
   * associated with the action this key is bound to.
   */
  public static final String PERSISTENT_WORKER_KEY = "persistentWorkerKey";

  /**
   * (nonstandard) The required protocol for remote persistent workers.
   *
   * <p>This communicates to the remote execution server which protocol a persistent worker is
   * expecting for the action this platform property is bound to.
   *
   * <p>Supported values: json, protobuf.
   */
  public static final String PERSISTENT_WORKER_PROTOCOL = "persistentWorkerProtocol";
}
