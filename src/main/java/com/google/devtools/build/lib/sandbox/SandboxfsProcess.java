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

import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;

/** Interface to interact with a sandboxfs instance. */
interface SandboxfsProcess {

  /** Represents a single mapping within a sandboxfs file system. */
  @AutoValue
  abstract class Mapping {
    /**
     * Path within the sandbox.  This looks like an absolute path but is treated as relative to the
     * sandbox's root.
     */
    abstract PathFragment path();

    /** Absolute path from the host's file system to map into the sandbox. */
    abstract PathFragment target();

    /** Whether the mapped path is writable or not. */
    abstract boolean writable();

    /** Constructs a new mapping builder. */
    static Builder builder() {
      return new AutoValue_SandboxfsProcess_Mapping.Builder();
    }

    /** Builder for a single mapping within a sandboxfs file system. */
    @AutoValue.Builder
    abstract static class Builder {
      /**
       * Sets the path within the sandbox on which this mapping will appear.  This looks like an
       * absolute path but is treated as relative to the sandbox's root.
       *
       * @param path absolute path rooted at the sandbox's mount point
       * @return the builder instance
       */
      abstract Builder setPath(PathFragment path);

      /**
       * Sets the path to which this mapping refers.  This is an absolute path into the host's
       * file system.
       *
       * @param target absolute path into the host's file system
       * @return the builder instance
       */
      abstract Builder setTarget(PathFragment target);

      /**
       * Sets whether this mapping is writable or not when accessed via the sandbox.
       *
       * @param writable whether the mapping is writable or not
       * @return the builder instance
       */
      abstract Builder setWritable(boolean writable);

      abstract Mapping autoBuild();

      /**
       * Constructs the mapping and validates field invariants.
       *
       * @return the constructed mapping.
       */
      public Mapping build() {
        Mapping mapping = autoBuild();
        checkState(mapping.path().isAbsolute(), "Mapping specifications are supposed to be "
            + "absolute but %s is not", mapping.path());
        checkState(mapping.target().isAbsolute(), "Mapping targets are supposed to be "
            + "absolute but %s is not", mapping.target());
        return mapping;
      }
    }
  }

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

  /**
   * Adds new mappings to the sandboxfs instance.
   *
   * @param mappings the collection of mappings to add, which must not have yet been previously
   *     mapped
   * @throws IOException if sandboxfs cannot be reconfigured either because of an error in the
   *     configuration or because we failed to communicate with the subprocess
   */
  void map(List<Mapping> mappings) throws IOException;

  /**
   * Removes a mapping from the sandboxfs instance.
   *
   * @param mapping the mapping to remove, which must have been previously mapped.  This looks like
   *     an absolute path but is treated as relative to the sandbox's root.
   * @throws IOException if sandboxfs cannot be reconfigured either because of an error in the
   *     configuration or because we failed to communicate with the subprocess
   */
  void unmap(PathFragment mapping) throws IOException;
}
