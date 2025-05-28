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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.packages.Package.Builder.PackageLimits;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.Package.ConfigSettingVisibilityPolicy;
import com.google.devtools.build.lib.packages.Package.Declarations;
import com.google.devtools.build.lib.packages.Package.Metadata;
import com.google.devtools.build.lib.packages.TargetRecorder.MacroNamespaceViolationException;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.Optional;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.syntax.Location;

/**
 * A piece of a {@link Package}: either the subset obtained by evaluating a BUILD file and not
 * expanding any symbolic macros; or the subset obtained by evaluating exactly one symbolic macro
 * instance.
 *
 * <p>To obtain a {@link Package} from a {@link PackagePiece}, use a PackageProvider or skyframe
 * machinery.
 */
// TODO(https://github.com/bazelbuild/bazel/issues/23852): as a future optimization, consider adding
// another class of package piece obtained by evaluating a set of macros.
public abstract sealed class PackagePiece extends Packageoid
    permits PackagePiece.ForBuildFile, PackagePiece.ForMacro {
  /**
   * The collection of all symbolic macro instances defined in this package piece, indexed by their
   * name (not by {@link MacroInstance#getId id} - contrast with {@link Package#macros}). Null until
   * the package piece is fully initialized by {@link #setMacrosByName}, in turn called by this
   * package piece's builder's {@code finishBuild()}.
   */
  @Nullable private ImmutableSortedMap<String, MacroInstance> macrosByName;

  public abstract PackagePieceIdentifier getIdentifier();

  /**
   * Returns a (read-only, ordered) iterable of all the targets belonging to this package piece
   * which are instances of the specified class. Doesn't search in any other package pieces.
   */
  public <T extends Target> Iterable<T> getTargets(Class<T> targetClass) {
    return Iterables.filter(targets.values(), targetClass);
  }

  @Override
  public Target getTarget(String targetName) throws NoSuchTargetException {
    Target target = targets.get(targetName);
    if (target != null) {
      return target;
    }

    throw noSuchTargetException(targetName);
  }

  /**
   * Returns the macro instance declared in this package piece having the provided name; or null if
   * no such macro instance exists.
   */
  @Nullable
  public MacroInstance getMacroByName(String name) {
    return macrosByName.get(name);
  }

  private NoSuchTargetException noSuchTargetException(String targetName) {
    Label label;
    try {
      label = Label.create(getPackageIdentifier(), targetName);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(targetName, e);
    }

    if (getMetadata().succinctTargetNotFoundErrors()) {
      return new NoSuchTargetException(
          label,
          String.format("target '%s' not declared in %s", targetName, getShortDescription()));
    } else {
      String alternateTargetSuggestion =
          Package.getAlternateTargetSuggestion(getMetadata(), targetName, targets.keySet());
      return new NoSuchTargetException(
          label,
          String.format(
              "target '%s' not declared in %s%s",
              targetName, getShortDescription(), alternateTargetSuggestion));
    }
  }

  @Override
  public String toString() {
    return String.format(
        "PackagePiece(%s defined by %s)=%s",
        getIdentifier().getCanonicalFormName(),
        getCanonicalFormDefinedBy(),
        targets != null ? getTargets(Rule.class) : "initializing...");
  }

  /**
   * Returns the canonical form of the BUILD file label if this is a {@link
   * PackagePiece.ForBuildFile}, or the canonical form of the macro class's declaring .bzl label and
   * macro name, in {@code label%name} format, if this is a {@link PackagePiece.ForMacro}.
   */
  public abstract String getCanonicalFormDefinedBy();

  /**
   * Sets the macros map for this package piece. Intended only to be called by this package piece's
   * builder.
   *
   * @param macros a collection of macro instances, which must have unique names.
   */
  protected void setMacrosByName(Collection<MacroInstance> macros) {
    ImmutableSortedMap.Builder<String, MacroInstance> macrosByName =
        ImmutableSortedMap.naturalOrder();
    for (MacroInstance macro : macros) {
      macrosByName.put(macro.getName(), macro);
    }
    this.macrosByName = macrosByName.buildOrThrow();
  }

  protected PackagePiece(Metadata metadata, Declarations declarations) {
    super(metadata, declarations);
  }

  /**
   * A {@link PackagePiece} obtained by evaluating a BUILD file, without expanding any symbolic
   * macros.
   */
  public static final class ForBuildFile extends PackagePiece {
    private final PackagePieceIdentifier.ForBuildFile identifier;
    // Can be changed during BUILD file evaluation due to exports_files() modifying its visibility.
    // Cannot be in declarations because, since it's a Target, it holds a back reference to this
    // PackagePiece.ForBuildFile object.
    private InputFile buildFile;

    @Override
    public PackagePieceIdentifier.ForBuildFile getIdentifier() {
      return identifier;
    }

    @Override
    public String getCanonicalFormDefinedBy() {
      return getMetadata().buildFileLabel().getCanonicalForm();
    }

    @Override
    public String getShortDescription() {
      return String.format("top-level package piece defined by %s", getCanonicalFormDefinedBy());
    }

    /** Returns the InputFile target for this package's BUILD file. */
    public InputFile getBuildFile() {
      return buildFile;
    }

    @Override
    public void checkMacroNamespaceCompliance(Target target) {
      checkArgument(this.equals(target.getPackageoid()), "Target must belong to this packageoid");
      // No-op: no macros to violate.
    }

    private ForBuildFile(PackagePieceIdentifier.ForBuildFile identifier, Metadata metadata) {
      super(metadata, new Declarations());
      checkArgument(identifier.getPackageIdentifier().equals(metadata.packageIdentifier()));
      this.identifier = identifier;
    }

    /** Creates a new {@link PackagePiece.ForBuildFile.Builder}. */
    // TODO(bazel-team): when JEP 482 ("flexible constructors") is enabled, we can remove this
    // method and use the builder's constructor directly.
    public static Builder newBuilder(
        PackageSettings packageSettings,
        PackagePieceIdentifier.ForBuildFile identifier,
        RootedPath filename,
        String workspaceName,
        Optional<String> associatedModuleName,
        Optional<String> associatedModuleVersion,
        boolean noImplicitFileExport,
        boolean simplifyUnconditionalSelectsInRuleAttrs,
        RepositoryMapping repositoryMapping,
        RepositoryMapping mainRepositoryMapping,
        @Nullable Semaphore cpuBoundSemaphore,
        PackageOverheadEstimator packageOverheadEstimator,
        @Nullable ImmutableMap<Location, String> generatorMap,
        @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy,
        @Nullable Globber globber,
        boolean enableNameConflictChecking,
        boolean trackFullMacroInformation,
        PackageLimits packageLimits) {
      Metadata metadata =
          Metadata.builder()
              .packageIdentifier(identifier.getPackageIdentifier())
              .buildFilename(filename)
              .isRepoRulePackage(false)
              .repositoryMapping(repositoryMapping)
              .associatedModuleName(associatedModuleName)
              .associatedModuleVersion(associatedModuleVersion)
              .configSettingVisibilityPolicy(configSettingVisibilityPolicy)
              .succinctTargetNotFoundErrors(packageSettings.succinctTargetNotFoundErrors())
              .build();
      ForBuildFile forBuildFile = new ForBuildFile(identifier, metadata);
      return new Builder(
          forBuildFile,
          packageSettings.precomputeTransitiveLoads(),
          noImplicitFileExport,
          simplifyUnconditionalSelectsInRuleAttrs,
          workspaceName,
          mainRepositoryMapping,
          cpuBoundSemaphore,
          packageOverheadEstimator,
          generatorMap,
          globber,
          enableNameConflictChecking,
          trackFullMacroInformation,
          packageLimits);
    }

    /** A builder for {@link PackagePiece.ForBuildFile} objects. */
    public static class Builder extends Package.AbstractBuilder {

      /** Retrieves this object from a Starlark thread. Returns null if not present. */
      @Nullable
      public static Builder fromOrNull(StarlarkThread thread) {
        StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
        return ctx instanceof Builder builder ? builder : null;
      }

      public PackagePiece.ForBuildFile getPackagePiece() {
        return (PackagePiece.ForBuildFile) pkg;
      }

      @Override
      @CanIgnoreReturnValue
      public Builder setLoads(Iterable<Module> directLoads) {
        return (Builder) super.setLoads(directLoads);
      }

      @Override
      public boolean eagerlyExpandMacros() {
        return false;
      }

      @Override
      @CanIgnoreReturnValue
      public Builder buildPartial() throws NoSuchPackageException {
        return (Builder) super.buildPartial();
      }

      @Override
      protected void setBuildFile(InputFile buildFile) {
        ((ForBuildFile) pkg).buildFile = checkNotNull(buildFile);
      }

      @Override
      public ForBuildFile finishBuild() {
        return (ForBuildFile) super.finishBuild();
      }

      @Override
      protected void packageoidInitializationHook() {
        super.packageoidInitializationHook();
        getPackagePiece().computationSteps = getComputationSteps();
        getPackagePiece().setMacrosByName(recorder.getMacroMap().values());
      }

      private Builder(
          ForBuildFile forBuildFile,
          boolean precomputeTransitiveLoads,
          boolean noImplicitFileExport,
          boolean simplifyUnconditionalSelectsInRuleAttrs,
          String workspaceName,
          RepositoryMapping mainRepositoryMapping,
          @Nullable Semaphore cpuBoundSemaphore,
          PackageOverheadEstimator packageOverheadEstimator,
          @Nullable ImmutableMap<Location, String> generatorMap,
          @Nullable Globber globber,
          boolean enableNameConflictChecking,
          boolean trackFullMacroInformation,
          PackageLimits packageLimits) {
        super(
            forBuildFile.getMetadata(),
            forBuildFile,
            SymbolGenerator.create(forBuildFile.getIdentifier()),
            precomputeTransitiveLoads,
            noImplicitFileExport,
            simplifyUnconditionalSelectsInRuleAttrs,
            workspaceName,
            mainRepositoryMapping,
            cpuBoundSemaphore,
            packageOverheadEstimator,
            generatorMap,
            globber,
            enableNameConflictChecking,
            trackFullMacroInformation,
            /* enableTargetMapSnapshotting= */ false,
            packageLimits);
      }
    }
  }

  /** A {@link PackagePiece} obtained by evaluating a symbolic macro instance. */
  public static final class ForMacro extends PackagePiece {
    private final PackagePieceIdentifier.ForMacro identifier;
    private final MacroInstance evaluatedMacro;
    // Null until the package piece is fully initialized by its builder's {@code finishBuild()}.
    @Nullable private ImmutableSet<String> macroNamespaceViolations = null;

    @Override
    public PackagePieceIdentifier.ForMacro getIdentifier() {
      return identifier;
    }

    @Override
    public String getCanonicalFormDefinedBy() {
      MacroClass macroClass = evaluatedMacro.getMacroClass();
      return String.format(
          "%s%%%s", macroClass.getDefiningBzlLabel().getCanonicalForm(), macroClass.getName());
    }

    @Override
    public String getShortDescription() {
      return String.format(
          "package piece %s defined by %s",
          getIdentifier().getCanonicalFormName(), getCanonicalFormDefinedBy());
    }

    public MacroInstance getEvaluatedMacro() {
      return evaluatedMacro;
    }

    /**
     * Returns the ID of the package of the .bzl file declaring the macro which was expanded to
     * produce this package piece; it is considered to be the location in which this package piece's
     * targets are declared for visibility purposes.
     */
    public PackageIdentifier getDeclaringPackage() {
      return evaluatedMacro.getMacroClass().getDefiningBzlLabel().getPackageIdentifier();
    }

    @Override
    public void checkMacroNamespaceCompliance(Target target)
        throws MacroNamespaceViolationException {
      checkArgument(this.equals(target.getPackageoid()), "Target must belong to this packageoid");
      checkNotNull(
          macroNamespaceViolations,
          "This method is only available after the package piece has been fully initialized.");
      if (macroNamespaceViolations.contains(target.getName())) {
        throw new MacroNamespaceViolationException(
            String.format(
                "Target %s declared in symbolic macro '%s' violates macro naming rules and cannot"
                    + " be built. %s",
                target.getLabel(), evaluatedMacro.getName(), TargetRecorder.MACRO_NAMING_RULES));
      }
    }

    private static void checkIdentifierMatchesMacro(
        PackagePieceIdentifier.ForMacro identifier, MacroInstance macro) {
      checkArgument(
          macro.getPackageMetadata().packageIdentifier().equals(identifier.getPackageIdentifier()));
      checkArgument(macro.getName().equals(identifier.getInstanceName()));
    }

    private ForMacro(
        Metadata metadata,
        Declarations declarations,
        MacroInstance evaluatedMacro,
        PackagePieceIdentifier parentIdentifier) {
      super(metadata, declarations);
      checkArgument(
          metadata
              .packageIdentifier()
              .equals(evaluatedMacro.getPackageMetadata().packageIdentifier()));
      checkArgument(metadata.packageIdentifier().equals(parentIdentifier.getPackageIdentifier()));
      if (evaluatedMacro.getParent() != null) {
        checkIdentifierMatchesMacro(
            (PackagePieceIdentifier.ForMacro) parentIdentifier, evaluatedMacro.getParent());
      } else {
        checkArgument(parentIdentifier instanceof PackagePieceIdentifier.ForBuildFile);
      }
      this.identifier =
          new PackagePieceIdentifier.ForMacro(
              metadata.packageIdentifier(), parentIdentifier, evaluatedMacro.getName());
      this.evaluatedMacro = evaluatedMacro;
    }

    /** Creates a new {@link PackagePiece.ForMacro.Builder}. */
    // TODO(bazel-team): when JEP 482 ("flexible constructors") is enabled, we can remove this
    // method and use the builder's constructor directly.
    public static Builder newBuilder(
        Metadata metadata,
        Declarations declarations,
        MacroInstance evaluatedMacro,
        PackagePieceIdentifier parentIdentifier,
        boolean simplifyUnconditionalSelectsInRuleAttrs,
        RepositoryMapping mainRepositoryMapping,
        @Nullable Semaphore cpuBoundSemaphore,
        PackageOverheadEstimator packageOverheadEstimator,
        boolean enableNameConflictChecking,
        boolean trackFullMacroInformation,
        PackageLimits packageLimits) {
      ForMacro forMacro = new ForMacro(metadata, declarations, evaluatedMacro, parentIdentifier);
      return new Builder(
          forMacro,
          simplifyUnconditionalSelectsInRuleAttrs,
          mainRepositoryMapping,
          cpuBoundSemaphore,
          packageOverheadEstimator,
          enableNameConflictChecking,
          trackFullMacroInformation,
          packageLimits);
    }

    /** A builder for {@link PackagePieceForMacro} objects. */
    public static class Builder extends TargetDefinitionContext {

      /** Retrieves this object from a Starlark thread. Returns null if not present. */
      @Nullable
      public static Builder fromOrNull(StarlarkThread thread) {
        StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
        return ctx instanceof Builder builder ? builder : null;
      }

      public PackagePiece.ForMacro getPackagePiece() {
        return (PackagePiece.ForMacro) pkg;
      }

      @Override
      public boolean eagerlyExpandMacros() {
        return false;
      }

      @Override
      @CanIgnoreReturnValue
      public Builder buildPartial() throws NoSuchPackageException {
        return (Builder) super.buildPartial();
      }

      @Override
      public ForMacro finishBuild() {
        return (ForMacro) super.finishBuild();
      }

      @Override
      protected void packageoidInitializationHook() {
        getPackagePiece().computationSteps = getComputationSteps();
        super.packageoidInitializationHook();
        ForMacro forMacro = getPackagePiece();
        forMacro.setMacrosByName(recorder.getMacroMap().values());
        forMacro.macroNamespaceViolations =
            ImmutableSet.copyOf(recorder.getMacroNamespaceViolatingTargets().keySet());
      }

      private Builder(
          ForMacro forMacro,
          boolean simplifyUnconditionalSelectsInRuleAttrs,
          RepositoryMapping mainRepositoryMapping,
          @Nullable Semaphore cpuBoundSemaphore,
          PackageOverheadEstimator packageOverheadEstimator,
          boolean enableNameConflictChecking,
          boolean trackFullMacroInformation,
          PackageLimits packageLimits) {
        super(
            forMacro.getMetadata(),
            forMacro,
            SymbolGenerator.create(forMacro.getIdentifier()),
            simplifyUnconditionalSelectsInRuleAttrs,
            forMacro.getDeclarations().getWorkspaceName(),
            mainRepositoryMapping,
            cpuBoundSemaphore,
            packageOverheadEstimator,
            /* generatorMap= */ null,
            /* globber= */ null,
            enableNameConflictChecking,
            trackFullMacroInformation,
            /* enableTargetMapSnapshotting= */ false,
            packageLimits);
      }
    }
  }
}
