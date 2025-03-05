// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.TargetRecorder.MacroNamespaceViolationException;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import javax.annotation.Nullable;

/**
 * A package-like entity (either a full {@link Package} or a package piece) which serves as a
 * container of {@link Target} and {@link MacroInstance} objects and links them to the package's
 * metadata and declarations.
 *
 * <p>Each {@link Target} or {@link MacroInstance} object is uniquely owned one packageoid (for
 * targets, that's the packageoid returned by {@link Target#getPackageoid()}). In some cases, the
 * target or macro instance may also be referenced from other packageoids, provided that this
 * doesn't create a skyframe cycle.
 *
 * <p>To obtain a {@link Package} from a {@link Packageoid}, use a PackageProvider or skyframe
 * machinery.
 */
public interface Packageoid {
  /**
   * Returns the metadata of the package; in other words, information which is known about a package
   * before BUILD file evaluation has started.
   */
  Package.Metadata getMetadata();

  /**
   * Returns this package's identifier. This is a convenience wrapper for {@link
   * Package.Metadata#packageIdentifier()}.
   */
  default PackageIdentifier getPackageIdentifier() {
    return getMetadata().packageIdentifier();
  }

  /**
   * Returns data about the package which is known after BUILD file evaluation without expanding
   * symbolic macros.
   */
  Package.Declarations getDeclarations();

  /**
   * Returns true if errors were encountered during evaluation of this packageoid. (The packageoid
   * may be incomplete and its contents should not be relied upon for critical operations. However,
   * any Rules belonging to the packageoid are guaranteed to be intact, unless their {@code
   * containsErrors()} flag is set.)
   */
  boolean containsErrors();

  /**
   * Returns the first {@link FailureDetail} describing one of the packageoid's errors, or {@code
   * null} if it has no errors or all its errors lack details.
   */
  @Nullable
  FailureDetail getFailureDetail();

  /**
   * Throws {@link MacroNamespaceViolationException} if the given target (which must be a member of
   * this packageoid) violates macro naming rules.
   */
  void checkMacroNamespaceCompliance(Target target) throws MacroNamespaceViolationException;

  /**
   * Returns the target (a member of this packagoid) whose name is "targetName". First rules are
   * searched, then output files, then input files. The target name must be valid, as defined by
   * {@code LabelValidator#validateTargetName}.
   *
   * <p>Use with care. In particular, note that {@code target.getPackageoid().getTarget("sibling")}
   * will succeed for all package-wide sibling targets if the packageoid is a package, but will
   * throw for targets belonging to a different package piece if the packageoid is a package piece.
   *
   * @throws NoSuchTargetException if the specified target was not found in this packageoid.
   */
  Target getTarget(String targetName) throws NoSuchTargetException;
}
