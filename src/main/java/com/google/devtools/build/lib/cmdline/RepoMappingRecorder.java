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

package com.google.devtools.build.lib.cmdline;

import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Records the lookups through repo mappings performed by the evaluation running on the current
 * thread.
 *
 * <p>Lookups through a {@link RepositoryMapping} that {@linkplain
 * RepositoryMapping#withRecordedLookups() records lookups} are reported to the recorder {@linkplain
 * #install installed} on the current thread, if any.
 */
@FunctionalInterface
public interface RepoMappingRecorder {

  /** Records that {@code apparentName} resolved to {@code canonicalName} in {@code mapping}. */
  void record(RepositoryMapping mapping, String apparentName, RepositoryName canonicalName);

  /**
   * Returns the canonical name of {@code apparentName} in {@code mapping} and records the lookup.
   * Recorders that know a more recent answer than {@code mapping} may return that instead.
   */
  default RepositoryName get(RepositoryMapping mapping, String apparentName) {
    RepositoryName canonicalName = mapping.getUnrecorded(apparentName);
    record(mapping, apparentName, canonicalName);
    return canonicalName;
  }

  /**
   * Returns the first apparent name of {@code canonicalName} in {@code mapping}. Recorders that
   * track inverse lookups override this.
   */
  default Optional<String> getInverse(RepositoryMapping mapping, RepositoryName canonicalName) {
    return mapping.getInverseUnrecorded(canonicalName);
  }

  /** Restores the previously installed recorder when closed. */
  interface Installation extends AutoCloseable {
    @Override
    void close();
  }

  /** Installs {@code recorder} on the current thread until the returned installation is closed. */
  static Installation install(@Nullable RepoMappingRecorder recorder) {
    RepoMappingRecorder previous = Holder.current.get();
    Holder.current.set(recorder);
    return () -> Holder.current.set(previous);
  }

  /** Returns the recorder installed on the current thread, if any. */
  @Nullable
  static RepoMappingRecorder current() {
    return Holder.current.get();
  }

  /** Holds the thread-local state of {@link RepoMappingRecorder}. */
  final class Holder {
    private static final ThreadLocal<RepoMappingRecorder> current = new ThreadLocal<>();

    private Holder() {}
  }
}
