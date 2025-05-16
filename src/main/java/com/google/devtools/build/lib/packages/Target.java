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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A node in the build dependency graph, identified by a Label.
 *
 * <p>This StarlarkBuiltin does not contain any documentation since Starlark's Target type refers to
 * TransitiveInfoCollection.class, which contains the appropriate documentation.
 */
public interface Target extends TargetData {

  /** Returns the {@link Packageoid} to which this target belongs. */
  Packageoid getPackageoid();

  /**
   * If this target is a direct member of a full {@link Package}, returns it; otherwise, returns
   * null.
   *
   * <p>Avoid adding new uses of this method; it is incompatible with lazy symbolic macro
   * evaluation.
   */
  @Nullable
  default Package getPackage() {
    return getPackageoid() instanceof Package ? (Package) getPackageoid() : null;
  }

  /** Returns the Package.Metadata of the package to which this target belongs. */
  // Overlaps signature of RuleOrMacroInstance#getPackageMetadata.
  Package.Metadata getPackageMetadata();

  /** Returns the Package.Declarations of the package to which this target belongs. */
  Package.Declarations getPackageDeclarations();

  /**
   * Returns the innermost symbolic macro that declared this target, or null if it was declared
   * outside any symbolic macro (i.e. directly in a BUILD file or only in one or more legacy
   * macros).
   *
   * <p>For targets in deserialized packages, throws {@link IllegalStateException}.
   */
  // Overlaps RuleorMacroInstance#getDeclaringMacro.
  @Nullable
  default MacroInstance getDeclaringMacro() {
    Packageoid packageoid = getPackageoid();
    if (packageoid instanceof Package pkg) {
      return pkg.getDeclaringMacroForTarget(getName());
      // TODO: #19922 - We might replace Package#getDeclaringMacroForTarget by storing a reference
      // to the declaring macro in implementations of this interface (sharing memory with the field
      // for the package).
    } else if (packageoid instanceof PackagePiece.ForMacro forMacro) {
      return forMacro.getEvaluatedMacro();
    } else if (packageoid instanceof PackagePiece.ForBuildFile) {
      return null;
    } else {
      throw new AssertionError("Unknown packageoid " + packageoid);
    }
  }

  /**
   * Returns the package that is considered to be the declaring location of this target.
   *
   * <p>For targets created inside a symbolic macro, this is the package containing the .bzl code of
   * the innermost running symbolic macro. For targets not in any symbolic macro, this is the same
   * as the package the target lives in.
   */
  // TODO(bazel-team): Clean up terminology throughout Target, RuleOrMacroInstance, MacroInstance,
  // CommonPrerequisiteInvalidator, etc., to be consistent. "Definition location" is the place where
  // a macro's .bzl code lives. "Declaration location" is the place where a target or macro instance
  // has its visibility checked (assuming no delegation applies) -- the definition location of its
  // declaring macro, or the BUILD file if not in a macro. "Declaring package" is perhaps ambiguous
  // and could mean either the declaration location or the package the target lives in.
  default PackageIdentifier getDeclaringPackage() {
    Packageoid packageoid = getPackageoid();
    if (packageoid instanceof Package pkg) {
      PackageIdentifier pkgId = pkg.getDeclaringPackageForTargetIfInMacro(getName());
      return pkgId != null ? pkgId : pkg.getPackageIdentifier();
    } else if (packageoid instanceof PackagePiece.ForMacro forMacro) {
      return forMacro.getDeclaringPackage();
    } else if (packageoid instanceof PackagePiece.ForBuildFile forBuildFile) {
      return forBuildFile.getPackageIdentifier();
    } else {
      throw new AssertionError("Unknown packageoid " + packageoid);
    }
  }

  /**
   * Returns true if this target was declared within one or more symbolic macros, or false if it was
   * the product of running only a BUILD file and the legacy macros it called.
   */
  default boolean isCreatedInSymbolicMacro() {
    Packageoid packageoid = getPackageoid();
    if (packageoid instanceof Package pkg) {
      return pkg.getDeclaringPackageForTargetIfInMacro(getName()) != null;
    } else if (packageoid instanceof PackagePiece.ForMacro) {
      return true;
    } else if (packageoid instanceof PackagePiece.ForBuildFile) {
      return false;
    } else {
      throw new AssertionError("Unknown packageoid " + packageoid);
    }
  }

  /**
   * Returns the rule associated with this target, if any.
   *
   * <p>If this is a Rule, returns itself; it this is an OutputFile, returns its generating rule; if
   * this is an input file, returns null.
   */
  @Nullable
  Rule getAssociatedRule();

  /**
   * Returns the license associated with this target.
   */
  License getLicense();

  /**
   * Returns the visibility that was supplied at the point of this target's declaration -- e.g. the
   * {@code visibility} attribute/argument for a rule target or {@code exports_files()} declaration)
   * -- or null if none was given.
   *
   * <p>Although this value is "raw", it is still normalized through {@link
   * RuleVisibility#validateAndSimplify RuleVisibility parsing}, e.g. eliminating redundant {@code
   * //visibility:private} items and replacing the list with a single {@code //visibility:public}
   * item if at least one such item appears.
   *
   * <p>This value may be useful to tooling that wants to introspect a target's visibility via
   * {@code bazel query} and feed the result back into a modified target declaration, without
   * picking up the package's default visibility, or the added location of the package or symbolic
   * macro the target was declared in. It is not useful as a direct input to the visibility
   * semantics; for that see {@link #getActualVisibility}.
   *
   * <p>This is also the value that is introspected through {@code native.existing_rules()}, except
   * that null is replaced by an empty visibility.
   */
  @Nullable
  RuleVisibility getRawVisibility();

  /**
   * Returns the default visibility value to fall back on if this target does not have a raw
   * visibility.
   *
   * <p>Usually this is just the package's default visibility value for targets not declared in
   * symbolic macros, and private for targets within symbolic macros. (In other words, a package's
   * default visibility does not propagate to within a symbolic macro.) However, some targets may
   * inject additional default visibility behavior here.
   */
  default RuleVisibility getDefaultVisibility() {
    return isCreatedInSymbolicMacro()
        ? RuleVisibility.PRIVATE
        : getPackageDeclarations().getPackageArgs().defaultVisibility();
  }

  /**
   * Returns the {@link #getRawVisibility raw visibility} of this target, falling back on a {@link
   * #getDefaultVisibility default value} if no raw visibility was supplied.
   *
   * <p>Due to the fallback, the result cannot be null.
   *
   * <p>This value may be useful for introspecting a target's visibility and reporting it in a
   * context where the package's default visibility is not known. It is not useful as a direct input
   * to the visibility semantics; for that see {@link #getActualVisibility}.
   */
  // TODO(brandjon): Perhaps the default value within a symbolic macro should be the value of the
  // `--default_visibility` flag / PrecomputedValue. This would ensure targets within macros are
  // always visible within unit tests or escape-hatched builds.
  // TODO(jhorvitz): Usually one of the following two methods suffice. Try to remove this.
  default RuleVisibility getVisibility() {
    RuleVisibility result = getRawVisibility();
    return result != null ? result : getDefaultVisibility();
  }

  /**
   * Equivalent to calling {@link RuleVisibility#getDependencyLabels} on the value returned by
   * {@link #getVisibility}, but potentially more efficient.
   *
   * <p>Prefer this method over {@link #getVisibility} when only the dependency labels are needed
   * and not a {@link RuleVisibility} instance.
   */
  default Iterable<Label> getVisibilityDependencyLabels() {
    return getVisibility().getDependencyLabels();
  }

  /**
   * Equivalent to calling {@link RuleVisibility#getDeclaredLabels} on the value returned by {@link
   * #getVisibility}, but potentially more efficient.
   *
   * <p>Prefer this method over {@link #getVisibility} when only the declared labels are needed and
   * not a {@link RuleVisibility} instance.
   */
  default List<Label> getVisibilityDeclaredLabels() {
    return getVisibility().getDeclaredLabels();
  }

  /**
   * Returns the visibility of this target, as understood by the visibility semantics.
   *
   * <p>This is the result of {@link #getVisibility} unioned with the package where this target was
   * instantiated (which can differ from the package where this target lives if the target was
   * created inside a symbolic macro).
   *
   * <p>This is the value that feeds into visibility checking in the analysis phase. See {@link
   * ConfiguredTargetFactory#convertVisibility} and {@link
   * CommonPrerequisiteValidator#isVisibleToLocation}.
   */
  default RuleVisibility getActualVisibility() {
    RuleVisibility visibility = getVisibility();
    MacroInstance declaringMacro = getDeclaringMacro();
    PackageIdentifier instantiatingLoc =
        declaringMacro == null
            ? getPackageMetadata().packageIdentifier()
            : declaringMacro.getDefinitionPackage();
    return visibility.concatWithPackage(instantiatingLoc);
  }

  /** Returns whether this target type can be configured (e.g. accepts non-null configurations). */
  boolean isConfigurable();

  /**
   * Creates a compact representation of this target with enough information for dependent parents.
   */
  TargetData reduceForSerialization();

  /** Returns the label identifying as a string formatted for display. */
  default String getDisplayFormLabel() {
    return getLabel()
        .getDisplayForm(
            getLabel().getRepository().isMain() ? getPackageMetadata().repositoryMapping() : null);
  }
}
