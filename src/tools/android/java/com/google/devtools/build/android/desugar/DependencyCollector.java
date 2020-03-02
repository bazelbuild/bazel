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
package com.google.devtools.build.android.desugar;

import javax.annotation.Nullable;

/**
 * Interface for collecting desugaring metadata that we can use to double-check correct desugaring
 * at the binary level by looking at the metadata written for all Jars on the runtime classpath
 * (b/65645388). Use {@link NoWriteCollectors} for "no-op" collectors and {@link
 * com.google.devtools.build.android.desugar.dependencies.MetadataCollector} for writing out
 * metadata files.
 */
// TODO(kmb): There could conceivably be a "self-contained" version where we check at the end that
// we actually saw all the companion classes (in recordDefaultMethods) that we "assumed"; useful
// for one-shot runs over an entire binary.
@SuppressWarnings("unused") // default implementations consist of empty method stubs
public interface DependencyCollector {

  /** Class name suffix used for interface companion classes. */
  public String INTERFACE_COMPANION_SUFFIX = "$$CC";

  /**
   * Records that {@code origin} depends on companion class {@code target}. For the resulting binary
   * to be valid, {@code target} needs to exist, which isn't the case if the corresponding interface
   * is only available as a compile-time ("neverlink") dependency.
   */
  default void assumeCompanionClass(String origin, String target) {}

  /**
   * Records that {@code origin} transitively implements {@code target} but {@code target} isn't in
   * the classpath. This can lead to wrong desugarings if {@code target} or an interface it extends
   * defines default methods.
   */
  default void missingImplementedInterface(String origin, String target) {}

  /**
   * Records that the given interface extends the given interfaces.
   *
   * <p>This information is useful reference to double-check {@link #missingImplementedInterface}s
   * without reading and parsing .class files, specifically if default methods are defined in
   * interfaces that a missing interface transitively extends.
   */
  default void recordExtendedInterfaces(String origin, String... targets) {}

  /**
   * Records that the given interface has a companion class that includes the given number of
   * default methods (0 if there were only static methods). This method should not be called for
   * purely abstract interfaces, to allow verifying available companion classes against this.
   *
   * <p>This information is useful reference to double-check {@link #missingImplementedInterface}s
   * without reading and parsing .class files with better precision than just looking for companion
   * classes on the runtime classpath (which may only contain static methods).
   */
  default void recordDefaultMethods(String origin, int count) {}

  /**
   * Returns metadata to include into the desugaring output or {@code null} if none. Returning
   * anything but {@code null} will cause an extra file to be written into the output, including an
   * empty array.
   */
  @Nullable
  public byte[] toByteArray();

  /** Simple collectors that don't collect any information. */
  public enum NoWriteCollectors implements DependencyCollector {
    /** Singleton instance that does nothing. */
    NOOP,
    /**
     * Singleton instance that does nothing besides throwing if {@link #missingImplementedInterface}
     * is called.
     */
    FAIL_ON_MISSING {
      @Override
      public void missingImplementedInterface(String origin, String target) {
        throw new IllegalStateException(
            String.format(
                "Couldn't find interface %s on the classpath for desugaring %s", target, origin));
      }
    };

    @Override
    @Nullable
    public final byte[] toByteArray() {
      return null;
    }
  }
}
