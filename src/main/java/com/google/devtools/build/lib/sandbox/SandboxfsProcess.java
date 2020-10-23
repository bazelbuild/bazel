// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;

/** Interface to interact with a sandboxfs instance. */
interface SandboxfsProcess {

  /** Returns the path to the sandboxfs's mount point. */
  Path getMountPoint();

  /** Returns true if the sandboxfs process is still alive. */
  boolean isAlive();

  /**
   * Unmounts and stops the sandboxfs process.
   *
   * <p>This function must be idempotent because there can be a race between explicit calls during
   * regular execution and calls from shutdown hooks.
   */
  void destroy();

  /** Interface to create a single mapping definition within the sandbox. */
  @FunctionalInterface
  interface SandboxMapper {
    /**
     * Defines a single mapping within the sandbox being constructed.
     *
     * <p>Called only from within the context of a {@link SandboxCreator} instance, which carries
     * the details of the specific sandbox being constructed.
     *
     * @param path the path within the sandbox to define as a mapping. sandboxfs expects this path
     *     to be absolute.
     * @param underlyingPath the path to which the mapping points to, which is outside of the
     *     sandbox. sandboxfs expects this path to be absolute.
     * @param writable whether the mapping is writable or read-only
     * @throws IOException if the attempt to send the mapping to sandboxfs fails
     */
    void map(PathFragment path, PathFragment underlyingPath, boolean writable) throws IOException;
  }

  /** Interface to create all the mappings of a sandbox. */
  @FunctionalInterface
  interface SandboxCreator {
    /**
     * Creates all the mappings of a sandbox.
     *
     * <p>This lambda runs holding a big lock around sandboxfs and is called in the critical path of
     * all running actions. As a result, this lambda should not block (which includes not doing any
     * I/O other than writing to the sandboxfs stream via {@code mapper}).
     *
     * @param mapper a callback to send a single mapping definition to sandboxfs
     * @throws IOException if calls to the mapper fail, or if the lambda desires to raise any other
     *     problem during the construction of the sandbox
     */
    void create(SandboxMapper mapper) throws IOException;
  }

  /**
   * Creates a top-level directory with the given name and delegates the set up of all mappings
   * within that directory to the given {@code creator} lambda.
   *
   * @param name basename of the top-level directory to create
   * @param creator a callback to populate the sandbox with mappings
   * @throws IOException if sandboxfs cannot be reconfigured either because of an error in the
   *     configuration or because we failed to communicate with the subprocess
   */
  void createSandbox(String name, SandboxCreator creator) throws IOException;

  /**
   * Destroys a top-level directory and all of its contents.
   *
   * @param name basename of the top-level directory to destroy
   * @throws IOException if sandboxfs cannot be reconfigured either because of an error in the
   *     configuration or because we failed to communicate with the subprocess
   */
  void destroySandbox(String name) throws IOException;
}
