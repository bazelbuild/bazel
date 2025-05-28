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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.TargetRecorder.MacroNamespaceViolationException;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import java.util.OptionalLong;
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
public abstract class Packageoid {
  /** Sentinel value for package overhead being empty. */
  protected static final long PACKAGE_OVERHEAD_UNSET = -1;

  // ==== General package metadata fields ====

  protected final Package.Metadata metadata;

  protected final Package.Declarations declarations;

  // ==== Common metadata fields ====

  /**
   * True iff this packageoid's Starlark files contained lexical or grammatical errors, or
   * experienced errors during evaluation, or semantic errors during the construction of any rule.
   *
   * <p>Note: A packageoid containing errors does not necessarily prevent a build; if all the rules
   * needed for a given build were constructed prior to the first error, the build may proceed.
   */
  protected boolean containsErrors = false;

  /**
   * The first detailed error encountered during this packageoid's construction and evaluation, or
   * {@code null} if there were no such errors or all its errors lacked details.
   */
  @Nullable protected FailureDetail failureDetail;

  protected long computationSteps = 0;

  /**
   * A rough approximation of the memory and general accounting costs associated with a loaded
   * packageoid. A value of -1 means it is unset. Stored as a long to take up less memory per pkg.
   */
  protected long packageOverhead = PACKAGE_OVERHEAD_UNSET;

  // ==== Common target and macro fields ====

  /**
   * The collection of all targets defined in this packageoid, indexed by name. Null until the
   * packageoid is fully initialized by its builder's {@code finishBuild()}.
   */
  // TODO(bazel-team): Clarify what this map contains when a rule and its output both share the same
  // name.
  @Nullable protected ImmutableSortedMap<String, Target> targets;

  /**
   * Returns the metadata of the package; in other words, information which is known about a package
   * before BUILD file evaluation has started.
   */
  public Package.Metadata getMetadata() {
    return metadata;
  }

  /**
   * Returns the package's identifier. This is a convenience wrapper for {@link
   * Package.Metadata#packageIdentifier()}.
   */
  public PackageIdentifier getPackageIdentifier() {
    return getMetadata().packageIdentifier();
  }

  /**
   * Returns data about the package which is known after BUILD file evaluation without expanding
   * symbolic macros.
   */
  public Package.Declarations getDeclarations() {
    return declarations;
  }

  /**
   * Returns the label for the package's BUILD file.
   *
   * <p>Typically, <code>getBuildFileLabel().getName().equals("BUILD")</code> -- though not
   * necessarily: data in a subdirectory of a test package may use a different filename to avoid
   * inadvertently creating a new package.
   */
  public Label getBuildFileLabel() {
    return getMetadata().buildFileLabel();
  }

  /**
   * Returns a short, lower-case description of this packageoid, e.g. for use in logging and error
   * messages.
   */
  public abstract String getShortDescription();

  /**
   * Returns an (immutable, ordered) view of all the targets belonging to this packageoid. Note that
   * if this packageoid is a package piece, this method does not search for targets in any other
   * package pieces.
   */
  public ImmutableSortedMap<String, Target> getTargets() {
    return targets;
  }

  /**
   * Returns true if errors were encountered during evaluation of this packageoid.
   *
   * <p>If a packageoid contains errors, it may be incomplete and its contents should not be relied
   * upon for critical operations. All rules in such a packageoid will have their {@link
   * Rule#containsErrors()} flag set to true.
   */
  public boolean containsErrors() {
    return containsErrors;
  }

  /**
   * Marks this packageoid as in error.
   *
   * <p>This method may only be called while the packageoid is being constructed. Intended only for
   * use by {@link Rule#reportError}, since its callers might not have access to the packageoid's
   * builder instance.
   *
   * @throws IllegalStateException if this packageoid has completed construction.
   */
  void setContainsErrors() {
    checkState(
        targets == null,
        "setContainsErrors() can only be called while the packageoid is being constructed");
    containsErrors = true;
  }

  /**
   * Returns the first {@link FailureDetail} describing one of the packageoid's errors, or {@code
   * null} if it has no errors or all its errors lack details.
   */
  @Nullable
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }

  /**
   * Returns the number of Starlark computation steps executed during the evaluation of this
   * packageoid.
   */
  public long getComputationSteps() {
    return computationSteps;
  }

  /** Returns package overhead as configured by the configured {@link PackageOverheadEstimator}. */
  public OptionalLong getPackageOverhead() {
    return packageOverhead == PACKAGE_OVERHEAD_UNSET
        ? OptionalLong.empty()
        : OptionalLong.of(packageOverhead);
  }

  /**
   * Throws {@link MacroNamespaceViolationException} if the given target (which must be a member of
   * this packageoid) violates macro naming rules.
   */
  public abstract void checkMacroNamespaceCompliance(Target target)
      throws MacroNamespaceViolationException;

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
  public abstract Target getTarget(String targetName) throws NoSuchTargetException;

  protected Packageoid(Package.Metadata metadata, Package.Declarations declarations) {
    this.metadata = checkNotNull(metadata);
    this.declarations = checkNotNull(declarations);
  }
}
