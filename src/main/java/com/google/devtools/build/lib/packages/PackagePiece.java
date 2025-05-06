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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.Package.ConfigSettingVisibilityPolicy;
import com.google.devtools.build.lib.packages.Package.Declarations;
import com.google.devtools.build.lib.packages.Package.Metadata;
import com.google.devtools.build.lib.packages.TargetRecorder.MacroNamespaceViolationException;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
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
  public abstract PackagePieceIdentifier getIdentifier();

  /**
   * Returns the {@link PackagePiece} corresponding to the evaluation of the BUILD file for this
   * package.
   */
  public abstract PackagePiece.ForBuildFile getPackagePieceForBuildFile();

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
   * Returns the target with a specified name, searching this package piece and the package piece
   * for the package's BUILD file. Returns null if the target is not found. The target name must be
   * valid, as defined by {@code LabelValidator#validateTargetName}.
   *
   * <p>Unlike {@link #getTarget}, this method does not throw an exception if the target is not
   * found.
   */
  @Nullable
  public Target tryGetTargetHereOrBuildFile(String targetName) {
    @Nullable Target target = targets.get(targetName);
    if (target == null && getPackagePieceForBuildFile() != this) {
      target = getPackagePieceForBuildFile().targets.get(targetName);
    }
    return target;
  }

  /**
   * Returns the outermost macro instance declared in this package piece having the provided name;
   * or null if no such macro instance exists.
   */
  @Nullable
  @VisibleForTesting
  MacroInstance getMacroByName(String name) {
    // Note that `macros` is keyed by macro IDs, not names.
    return macros.values().stream().filter(m -> m.getName().equals(name)).findFirst().orElse(null);
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
          String.format(
              "target '%s' not declared in package piece '%s'", targetName, getIdentifier()));
    } else {
      String alternateTargetSuggestion =
          Package.getAlternateTargetSuggestion(getMetadata(), targetName, targets.keySet());
      return new NoSuchTargetException(
          label,
          String.format(
              "target '%s' not declared in package piece %s%s",
              targetName, getIdentifier(), alternateTargetSuggestion));
    }
  }

  @Override
  public String toString() {
    return "PackagePiece("
        + getIdentifier()
        + ")="
        + (targets != null ? getTargets(Rule.class) : "initializing...");
  }

  @Override
  public String getShortDescription() {
    return "package piece " + getIdentifier();
  }

  /**
   * A {@link PackagePiece} obtained by evaluating a BUILD file, without expanding any symbolic
   * macros.
   */
  public static final class ForBuildFile extends PackagePiece {
    private final PackagePieceIdentifier.ForBuildFile identifier;
    private final Metadata metadata;
    private final Declarations declarations;
    // Can be changed during BUILD file evaluation due to exports_files() modifying its visibility.
    // Cannot be in declarations because, since it's a Target, it holds a back reference to this
    // PackagePiece.ForBuildFile object.
    private InputFile buildFile;

    @Override
    public PackagePieceIdentifier.ForBuildFile getIdentifier() {
      return identifier;
    }

    @Override
    public PackagePiece.ForBuildFile getPackagePieceForBuildFile() {
      return this;
    }

    @Override
    public Metadata getMetadata() {
      return metadata;
    }

    @Override
    public Package.Declarations getDeclarations() {
      return declarations;
    }

    @Override
    public InputFile getBuildFile() {
      return buildFile;
    }

    @Override
    public void checkMacroNamespaceCompliance(Target target) {
      checkArgument(this.equals(target.getPackageoid()), "Target must belong to this packageoid");
      // No-op: no macros to violate.
    }

    private ForBuildFile(PackagePieceIdentifier.ForBuildFile identifier, Metadata metadata) {
      checkArgument(identifier.getPackageIdentifier().equals(metadata.packageIdentifier()));
      checkArgument(identifier.getDefiningLabel().equals(metadata.buildFileLabel()));
      this.identifier = identifier;
      this.metadata = metadata;
      this.declarations = new Declarations();
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
        boolean trackFullMacroInformation) {
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
          trackFullMacroInformation);
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
      void setComputationSteps(long n) {
        ((PackagePiece) pkg).computationSteps = n;
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
          boolean trackFullMacroInformation) {
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
            /* enableTargetMapSnapshotting= */ false);
      }
    }
  }

  /** A {@link PackagePiece} obtained by evaluating a symbolic macro instance. */
  public static final class ForMacro extends PackagePiece {
    private final PackagePieceIdentifier.ForMacro identifier;
    private final MacroInstance evaluatedMacro;
    private final PackagePiece.ForBuildFile pieceForBuildFile;
    // Null until the package piece is fully initialized by its builder's {@code finishBuild()}.
    @Nullable private ImmutableSet<String> macroNamespaceViolations = null;

    @Override
    public PackagePieceIdentifier.ForMacro getIdentifier() {
      return identifier;
    }

    @Override
    public PackagePiece.ForBuildFile getPackagePieceForBuildFile() {
      return pieceForBuildFile;
    }

    @Override
    public Metadata getMetadata() {
      return pieceForBuildFile.getMetadata();
    }

    @Override
    public Declarations getDeclarations() {
      return pieceForBuildFile.getDeclarations();
    }

    @Override
    public InputFile getBuildFile() {
      return pieceForBuildFile.getBuildFile();
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

    private ForMacro(MacroInstance evaluatedMacro, PackagePiece.ForBuildFile pieceForBuildFile) {
      this.identifier =
          new PackagePieceIdentifier.ForMacro(
              pieceForBuildFile.getPackageIdentifier(),
              evaluatedMacro.getMacroClass().getDefiningBzlLabel(),
              /* definingSymbol= */ evaluatedMacro.getMacroClass().getName(),
              /* instanceName= */ evaluatedMacro.getName());
      this.evaluatedMacro = evaluatedMacro;
      this.pieceForBuildFile = pieceForBuildFile;
    }

    /** Creates a new {@link PackagePiece.ForMacro.Builder}. */
    // TODO(bazel-team): when JEP 482 ("flexible constructors") is enabled, we can remove this
    // method and use the builder's constructor directly.
    public static Builder newBuilder(
        MacroInstance evaluatedMacro,
        PackagePiece.ForBuildFile pieceForBuildFile,
        boolean simplifyUnconditionalSelectsInRuleAttrs,
        RepositoryMapping repositoryMapping,
        RepositoryMapping mainRepositoryMapping,
        @Nullable Semaphore cpuBoundSemaphore,
        PackageOverheadEstimator packageOverheadEstimator,
        @Nullable ImmutableMap<Location, String> generatorMap,
        boolean enableNameConflictChecking,
        boolean trackFullMacroInformation) {
      ForMacro forMacro = new ForMacro(evaluatedMacro, pieceForBuildFile);
      return new Builder(
          forMacro,
          simplifyUnconditionalSelectsInRuleAttrs,
          mainRepositoryMapping,
          cpuBoundSemaphore,
          packageOverheadEstimator,
          generatorMap,
          enableNameConflictChecking,
          trackFullMacroInformation);
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
      void setComputationSteps(long n) {
        ((PackagePiece) pkg).computationSteps = n;
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
        getPackagePiece().macroNamespaceViolations =
            ImmutableSet.copyOf(recorder.getMacroNamespaceViolatingTargets().keySet());
      }

      private Builder(
          ForMacro forMacro,
          boolean simplifyUnconditionalSelectsInRuleAttrs,
          RepositoryMapping mainRepositoryMapping,
          @Nullable Semaphore cpuBoundSemaphore,
          PackageOverheadEstimator packageOverheadEstimator,
          @Nullable ImmutableMap<Location, String> generatorMap,
          boolean enableNameConflictChecking,
          boolean trackFullMacroInformation) {
        super(
            forMacro.getMetadata(),
            forMacro,
            SymbolGenerator.create(forMacro.getIdentifier()),
            simplifyUnconditionalSelectsInRuleAttrs,
            forMacro.getPackagePieceForBuildFile().getDeclarations().getWorkspaceName(),
            mainRepositoryMapping,
            cpuBoundSemaphore,
            packageOverheadEstimator,
            generatorMap,
            /* globber= */ null,
            enableNameConflictChecking,
            trackFullMacroInformation,
            /* enableTargetMapSnapshotting= */ false);
      }
    }
  }
}
