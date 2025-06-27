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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoBuilder;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext.LoadGraphVisitor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.StarlarkThreadContext;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Package.Builder.PackageLimits;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.TargetRecorder.MacroNamespaceViolationException;
import com.google.devtools.build.lib.packages.TargetRecorder.NameConflictException;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;
import net.starlark.java.syntax.Location;

/**
 * A package, which is a container of {@link Rule}s, each of which contains a dictionary of named
 * attributes.
 *
 * <p>Package instances are intended to be immutable and for all practical purposes can be treated
 * as such. Note, however, that some member variables exposed via the public interface are not
 * strictly immutable, so until their types are guaranteed immutable we're not applying the
 * {@code @Immutable} annotation here.
 *
 * <p>This class should not be extended - it's only non-final for mocking!
 *
 * <p>When changing this class, make sure to make corresponding changes to serialization!
 */
@SuppressWarnings("JavaLangClash")
public class Package extends Packageoid {

  // TODO(bazel-team): This class and its builder are ginormous. Future refactoring work might
  // attempt to separate the concerns of:
  //   - instantiating targets/macros, adding them to the package, and accessing/indexing them
  //     afterwards
  //   - utility logical like validating names, checking for conflicts, etc.
  //   - tracking and enforcement of limits
  //   - machinery specific to external package / WORKSPACE / bzlmod

  // ==== Static fields and enums ====

  /**
   * How to enforce config_setting visibility settings.
   *
   * <p>This is a temporary setting in service of https://github.com/bazelbuild/bazel/issues/12669.
   * After enough depot cleanup, config_setting will have the same visibility enforcement as all
   * other rules.
   */
  public enum ConfigSettingVisibilityPolicy {
    /** Don't enforce visibility for any config_setting. */
    LEGACY_OFF,
    /** Honor explicit visibility settings on config_setting, else use //visibility:public. */
    DEFAULT_PUBLIC,
    /** Enforce config_setting visibility exactly the same as all other rules. */
    DEFAULT_STANDARD
  }

  /**
   * The "workspace name" of packages generated by Bzlmod to contain repo rules.
   *
   * <p>Normally, packages containing repo rules are differentiated from packages containing build
   * rules by the {@link PackageIdentifier}: The singular repo-rule-containing package is {@code
   * //external}. However, in Bzlmod, packages containing repo rules need to have meaningful {@link
   * PackageIdentifier}s, so there needs to be some other way to distinguish them from
   * build-rule-containing packages. We use the following magic string as the "workspace name" for
   * repo-rule-containing packages generated by Bzlmod.
   *
   * @see Metadata#isRepoRulePackage()
   */
  private static final String DUMMY_WORKSPACE_NAME_FOR_BZLMOD_PACKAGES = "__dummy_workspace_bzlmod";

  // Can be changed during BUILD file evaluation due to exports_files() modifying its visibility.
  // Cannot be in Declarations because, since it's a Target, it holds a back reference to this
  // Package object.
  private InputFile buildFile;

  // ==== Target and macro fields ====

  /**
   * The collection of all symbolic macro instances defined in this package, indexed by their {@link
   * MacroInstance#getId id} (not name). Null until the package is fully initialized by its
   * builder's {@code finishBuild()}.
   */
  // TODO(bazel-team): Consider enforcing that macro namespaces are "exclusive", meaning that target
  // names may only suffix a macro name when the target is created (transitively) within the macro.
  // This would be a major change that would break the (common) use case where a BUILD file
  // declares both "foo" and "foo_test".
  @Nullable private ImmutableSortedMap<String, MacroInstance> macros;

  /**
   * A map from names of targets declared in a symbolic macro which violate macro naming rules, such
   * as "lib%{name}-src.jar" implicit outputs in java rules, to the name of the macro instance where
   * they were declared.
   *
   * <p>Initialized by the builder in {@link Builder#finishBuild}.
   */
  @Nullable private ImmutableMap<String, String> macroNamespaceViolatingTargets;

  /**
   * A map from names of targets declared in a symbolic macro to the (innermost) macro instance
   * where they were declared. Omits targets not declared in symbolic macros.
   *
   * <p>Null for packages produced by deserialization.
   */
  // TODO: #19922 - If this field were made serializable (currently it's not), it would subsume
  // macroNamespaceViolatingTargets, since we can just map the target to its macro and then check
  // whether it is in the macro's namespace.
  //
  // TODO: #19922 - Don't maintain this extra map of all macro-instantiated targets. We have a
  // couple options:
  //   1) Have Target store a reference to its declaring MacroInstance directly. To avoid adding a
  //      field to that class (a not insignificant cost), we can merge it with the reference to its
  //      package: If we're not in a macro, we point to the package, and if we are, we point to the
  //      innermost macro, and hop to the MacroInstance to get a reference to the Package (or parent
  //      macro).
  //   2) To support lazy macro evaluation, we'll probably need a prefix trie in Package to find the
  //      macros whose namespaces contain the requested target name. For targets that respect their
  //      macro's namespace, we could just look them up in the trie. This assumes we already know
  //      whether the target is well-named, which we wouldn't if we got rid of
  //      macroNamespaceViolatingTargets.
  @Nullable private ImmutableMap<String, MacroInstance> targetsToDeclaringMacro;

  /**
   * A map from names of targets declared in a symbolic macro to the package where the macro that
   * declared it was defined, as per {@link MacroInstance#getDefinitionPackage}. Omits targets not
   * declared in symbolic macros.
   *
   * <p>Null for packages not produced by deserialization.
   */
  @Nullable private ImmutableMap<String, PackageIdentifier> targetsToDeclaringPackage;

  // ==== Constructor ====

  /**
   * Constructs a new (incomplete) Package instance. Intended only for use by {@link
   * Package.Builder}.
   *
   * <p>Packages and Targets refer to one another. Therefore, the builder needs to have a Package
   * instance on-hand before it can associate any targets with the package. The {@link
   * Package.Metadata} fields like the package's name must be known before that point, while other
   * fields are filled in only when the builder calls {@link Builder#finishBuild}.
   */
  // TODO(#19922): Better separate fields that must be known a priori from those determined through
  // BUILD evaluation.
  private Package(Metadata metadata) {
    super(metadata, new Declarations());
  }

  // ==== General package metadata accessors ====

  /**
   * Returns the name of this package. If this build is using external repositories then this name
   * may not be unique!
   */
  public String getName() {
    return metadata.getName();
  }

  /** Like {@link #getName}, but has type {@code PathFragment}. */
  public PathFragment getNameFragment() {
    return getPackageIdentifier().getPackageFragment();
  }

  /**
   * Returns the filename of the BUILD file which defines this package. The parent directory of the
   * BUILD file is the package directory.
   */
  public RootedPath getFilename() {
    return metadata.buildFilename();
  }

  /** Returns the directory containing the package's BUILD file. */
  public Path getPackageDirectory() {
    return metadata.getPackageDirectory();
  }

  /**
   * How to enforce visibility on <code>config_setting</code> See {@link
   * ConfigSettingVisibilityPolicy} for details.
   *
   * <p>Null for repo rule packages.
   */
  @Nullable
  public ConfigSettingVisibilityPolicy getConfigSettingVisibilityPolicy() {
    return metadata.configSettingVisibilityPolicy();
  }

  /** Convenience wrapper for {@link Declarations#getWorkspaceName} */
  public String getWorkspaceName() {
    // To support declarations mocking, use getDeclarations() instead of directly using the field.
    return getDeclarations().getWorkspaceName();
  }

  /** Returns the InputFile target for this package's BUILD file. */
  public InputFile getBuildFile() {
    return buildFile;
  }

  /** Convenience wrapper for {@link Declarations#getPackageArgs} */
  public PackageArgs getPackageArgs() {
    // To support declarations mocking, use getDeclarations() instead of directly using the field.
    return getDeclarations().getPackageArgs();
  }

  /** Convenience wrapper for {@link Declarations#getMakeEnvironment} */
  public ImmutableMap<String, String> getMakeEnvironment() {
    // To support declarations mocking, use getDeclarations() instead of directly using the field.
    return getDeclarations().getMakeEnvironment();
  }

  /**
   * Returns the root of the source tree beneath which this package's BUILD file was found, or
   * {@link Optional#empty} if this package was derived from a WORKSPACE file.
   *
   * <p>Assumes invariant: If non-empty, {@code
   * getSourceRoot().get().getRelative(packageId.getSourceRoot()).equals(getPackageDirectory())}
   */
  public Optional<Root> getSourceRoot() {
    return metadata.sourceRoot();
  }

  private static ImmutableList<Label> computeTransitiveLoads(Iterable<Module> directLoads) {
    Set<Label> loads = new LinkedHashSet<>();
    BazelModuleContext.visitLoadGraphRecursively(directLoads, loads::add);
    return ImmutableList.copyOf(loads);
  }

  // ==== Target and macro accessors ====

  /**
   * Returns a (read-only, ordered) iterable of all the targets belonging to this package which are
   * instances of the specified class.
   */
  public <T extends Target> Iterable<T> getTargets(Class<T> targetClass) {
    return Iterables.filter(targets.values(), targetClass);
  }

  /**
   * Returns the rule that corresponds to a particular BUILD target name. Useful for walking through
   * the dependency graph of a target. Fails if the target is not a Rule.
   */
  public Rule getRule(String targetName) {
    return (Rule) targets.get(targetName);
  }

  /**
   * Returns a map from names of targets declared in a symbolic macro which violate macro naming
   * rules, such as "lib%{name}-src.jar" implicit outputs in java rules, to the name of the macro
   * instance where they were declared.
   */
  ImmutableMap<String, String> getMacroNamespaceViolatingTargets() {
    Preconditions.checkNotNull(
        macroNamespaceViolatingTargets,
        "This method is only available after the package has been loaded.");
    return macroNamespaceViolatingTargets;
  }

  /**
   * Returns a map from names of targets declared in a symbolic macro to the package containing said
   * macro's .bzl code.
   */
  ImmutableMap<String, PackageIdentifier> getTargetsToDeclaringPackage() {
    if (targetsToDeclaringPackage != null) {
      return targetsToDeclaringPackage;
    } else {
      ImmutableMap.Builder<String, PackageIdentifier> result = ImmutableMap.builder();
      for (Map.Entry<String, MacroInstance> entry : targetsToDeclaringMacro.entrySet()) {
        result.put(entry.getKey(), entry.getValue().getDefinitionPackage());
      }
      return result.buildOrThrow();
    }
  }

  @Override
  public void checkMacroNamespaceCompliance(Target target) throws MacroNamespaceViolationException {
    Preconditions.checkArgument(
        this.equals(target.getPackage()), "Target must belong to this package");
    @Nullable
    String macroNamespaceViolated = getMacroNamespaceViolatingTargets().get(target.getName());
    if (macroNamespaceViolated != null) {
      throw new MacroNamespaceViolationException(
          String.format(
              "Target %s declared in symbolic macro '%s' violates macro naming rules and cannot be"
                  + " built. %s",
              target.getLabel(), macroNamespaceViolated, TargetRecorder.MACRO_NAMING_RULES));
    }
  }

  @Override
  public Target getTarget(String targetName) throws NoSuchTargetException {
    Target target = targets.get(targetName);
    if (target != null) {
      return target;
    }

    Label label;
    try {
      label = Label.create(metadata.packageIdentifier(), targetName);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(targetName, e);
    }

    if (metadata.succinctTargetNotFoundErrors()) {
      throw new NoSuchTargetException(
          label, String.format("target '%s' not declared in package '%s'", targetName, getName()));
    } else {
      String alternateTargetSuggestion =
          getAlternateTargetSuggestion(metadata, targetName, targets.keySet());
      throw new NoSuchTargetException(
          label,
          String.format(
              "target '%s' not declared in package '%s' defined by %s%s",
              targetName,
              getName(),
              metadata.buildFilename().asPath().getPathString(),
              alternateTargetSuggestion));
    }
  }

  static String getAlternateTargetSuggestion(
      Metadata metadata, String targetName, ImmutableSet<String> otherTargets) {
    // If there's a file on the disk that's not mentioned in the BUILD file,
    // produce a more informative error.  NOTE! this code path is only executed
    // on failure, which is (relatively) very rare.  In the common case no
    // stat(2) is executed.
    Path filename = metadata.getPackageDirectory().getRelative(targetName);
    if (!PathFragment.isNormalized(targetName) || "*".equals(targetName)) {
      // Don't check for file existence if the target name is not normalized
      // because the error message would be confusing and wrong. If the
      // targetName is "foo/bar/.", and there is a directory "foo/bar", it
      // doesn't mean that "//pkg:foo/bar/." is a valid label.
      // Also don't check if the target name is a single * character since
      // it's invalid on Windows.
      return "";
    } else if (filename.isDirectory()) {
      return "; however, a source directory of this name exists.  (Perhaps add "
          + "'exports_files([\""
          + targetName
          + "\"])' to "
          + getRepoRelativeBuildFilePathString(metadata)
          + ", or define a "
          + "filegroup?)";
    } else if (filename.exists()) {
      return "; however, a source file of this name exists.  (Perhaps add "
          + "'exports_files([\""
          + targetName
          + "\"])' to "
          + getRepoRelativeBuildFilePathString(metadata)
          + "?)";
    } else {
      return TargetSuggester.suggestTargets(targetName, otherTargets);
    }
  }

  private static String getRepoRelativeBuildFilePathString(Metadata metadata) {
    return metadata
        .packageIdentifier()
        .getPackageFragment()
        .getRelative(metadata.buildFilename().asPath().getBaseName())
        .getPathString();
  }

  /**
   * Returns all symbolic macros defined in the package, indexed by {@link MacroInstance#getId id}.
   *
   * <p>Note that {@code MacroInstance}s hold just the information known at the time a macro was
   * declared, even though by the time the {@code Package} is fully constructed we already have
   * fully evaluated these macros.
   */
  public ImmutableMap<String, MacroInstance> getMacrosById() {
    return macros;
  }

  /**
   * Returns the (innermost) symbolic macro instance that declared the given target, or null if the
   * target was not created in a symbolic macro.
   *
   * <p>Throws {@link IllegalArgumentException} if the given name is not a target in this package.
   *
   * <p>For packages produced by deserialization, this information is not available and {@code
   * IllegalStateException} is thrown.
   */
  @Nullable
  public MacroInstance getDeclaringMacroForTarget(String target) {
    Preconditions.checkState(
        targetsToDeclaringMacro != null,
        "Cannot retrieve MacroInstance information from deserialized packages");
    Preconditions.checkArgument(targets.containsKey(target), "unknown target '%s'", target);
    return targetsToDeclaringMacro.get(target);
  }

  /**
   * Returns the id of the package where the (innermost) macro that declared the given target was
   * defined (as per {@link MacroInstance#getDefinitionLocation}), or null if the target was not
   * created in a symbolic macro.
   *
   * <p>The caller should interpret a null result to mean that the declaration location of the
   * target is this package.
   *
   * <p>Throws {@link IllegalArgumentException} if the given name is not a target in this package.
   */
  @Nullable
  public PackageIdentifier getDeclaringPackageForTargetIfInMacro(String target) {
    Preconditions.checkArgument(targets.containsKey(target), "unknown target '%s'", target);
    // Exactly one of targetsToDeclaringMacro and targetsToDeclaringPackage is non-null, depending
    // on whether this package was produced by deserialization.
    if (targetsToDeclaringMacro != null) {
      MacroInstance macro = targetsToDeclaringMacro.get(target);
      return macro != null ? macro.getDefinitionPackage() : null;
    } else {
      return targetsToDeclaringPackage.get(target);
    }
  }

  // ==== Stringification / debugging ====

  @Override
  public String toString() {
    return "Package("
        + getName()
        + ")="
        + (targets != null ? getTargets(Rule.class) : "initializing...");
  }

  @Override
  public String getShortDescription() {
    return "package " + getPackageIdentifier().getCanonicalForm();
  }

  /**
   * Dumps the package for debugging. Do not depend on the exact format/contents of this debugging
   * output.
   */
  public void dump(PrintStream out) {
    out.println("  Package " + getName() + " (" + metadata.buildFilename().asPath() + ")");

    // Rules:
    out.println("    Rules");
    for (Rule rule : getTargets(Rule.class)) {
      out.println("      " + rule.getTargetKind() + " " + rule.getLabel());
      for (Attribute attr : rule.getAttributes()) {
        for (Object possibleValue :
            AggregatingAttributeMapper.of(rule).visitAttribute(attr.getName(), attr.getType())) {
          out.println("        " + attr.getName() + " = " + possibleValue);
        }
      }
    }

    // Files:
    out.println("    Files");
    for (FileTarget file : getTargets(FileTarget.class)) {
      out.print("      " + file.getTargetKind() + " " + file.getLabel());
      if (file instanceof OutputFile) {
        out.println(" (generated by " + ((OutputFile) file).getGeneratingRule().getLabel() + ")");
      } else {
        out.println();
      }
    }
  }

  // ==== Error reporting ====

  /**
   * Returns an error {@link Event} with {@link Location} and {@link DetailedExitCode} properties.
   */
  public static Event error(Location location, String message, Code code) {
    return errorWithDetailedExitCode(
        location,
        message,
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setPackageLoading(PackageLoading.newBuilder().setCode(code))
                .build()));
  }

  /** Similar to {@link #error} but with a custom {@link DetailedExitCode}. */
  public static Event errorWithDetailedExitCode(
      Location location, String message, DetailedExitCode detailedExitCode) {
    Event error = Event.error(location, message);
    return error.withProperty(DetailedExitCode.class, detailedExitCode);
  }

  /**
   * If {@code pkg.containsErrors()}, sends an errorful "package contains errors" {@link Event}
   * (augmented with {@code pkg.getFailureDetail()}, if present) to the given {@link EventHandler}.
   */
  public static void maybeAddPackageContainsErrorsEventToHandler(
      Package pkg, EventHandler eventHandler) {
    if (pkg.containsErrors()) {
      eventHandler.handle(
          Event.error(
              String.format(
                  "package contains errors: %s%s",
                  pkg.getNameFragment(),
                  pkg.getFailureDetail() != null
                      ? ": " + pkg.getFailureDetail().getMessage()
                      : "")));
    }
  }

  /**
   * Given a {@link FailureDetail} and target, returns a modified {@code FailureDetail} that
   * attributes its error to the target.
   *
   * <p>If the given detail is null, then a generic {@link Code#TARGET_MISSING} detail identifying
   * the target is returned.
   */
  public static FailureDetail contextualizeFailureDetailForTarget(
      @Nullable FailureDetail failureDetail, Target target) {
    String prefix =
        "Target '" + target.getLabel() + "' contains an error and its package is in error";
    if (failureDetail == null) {
      return FailureDetail.newBuilder()
          .setMessage(prefix)
          .setPackageLoading(PackageLoading.newBuilder().setCode(Code.TARGET_MISSING))
          .build();
    }
    return failureDetail.toBuilder().setMessage(prefix + ": " + failureDetail.getMessage()).build();
  }

  // ==== Builders ====

  /**
   * Returns a new {@link Builder} suitable for constructing an ordinary package (i.e. not one for
   * WORKSPACE or bzlmod).
   */
  public static Builder newPackageBuilder(
      PackageSettings packageSettings,
      PackageIdentifier id,
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
      // TODO(bazel-team): See Builder() constructor comment about use of null for this param.
      @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy,
      @Nullable Globber globber,
      boolean enableNameConflictChecking,
      boolean trackFullMacroInformation,
      PackageLimits packageLimits) {
    // Determine whether this is for a repo rule package. We shouldn't actually have to do this
    // because newPackageBuilder() is supposed to only be called for normal packages. Unfortunately
    // serialization still uses the same code path for deserializing BUILD and WORKSPACE files,
    // violating this method's contract.
    boolean isRepoRulePackage =
        id.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)
            || workspaceName.equals(DUMMY_WORKSPACE_NAME_FOR_BZLMOD_PACKAGES);

    return new Builder(
        Metadata.builder()
            .packageIdentifier(id)
            .buildFilename(filename)
            .isRepoRulePackage(isRepoRulePackage)
            .repositoryMapping(repositoryMapping)
            .associatedModuleName(associatedModuleName)
            .associatedModuleVersion(associatedModuleVersion)
            .configSettingVisibilityPolicy(configSettingVisibilityPolicy)
            .succinctTargetNotFoundErrors(packageSettings.succinctTargetNotFoundErrors())
            .build(),
        SymbolGenerator.create(id),
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

  // Bzlmod creates one fake package per external repository. The repos created by a given
  // extension, which can be 1000s, share the same metadata.
  private static final Interner<Metadata> bzlmodMetadataInterner = BlazeInterners.newWeakInterner();

  public static Builder newExternalPackageBuilderForBzlmod(
      RootedPath moduleFilePath,
      boolean noImplicitFileExport,
      boolean simplifyUnconditionalSelectsInRuleAttrs,
      PackageIdentifier basePackageId,
      RepositoryMapping repoMapping) {
    // moduleFilePath is turned into a string and retained as the Location of
    // the created package. Ensure that this string is the same instance as
    // the one in the interned Metadata object.
    RootedPath absoluteRootModuleFilePath =
        RootedPath.toRootedPath(
            Root.absoluteRoot(moduleFilePath.getRoot().getFileSystem()), moduleFilePath.asPath());
    return new Builder(
            bzlmodMetadataInterner.intern(
                Metadata.builder()
                    .packageIdentifier(basePackageId)
                    .buildFilename(absoluteRootModuleFilePath)
                    .isRepoRulePackage(true)
                    .repositoryMapping(repoMapping)
                    .succinctTargetNotFoundErrors(
                        PackageSettings.DEFAULTS.succinctTargetNotFoundErrors())
                    .build()),
            SymbolGenerator.create(basePackageId),
            PackageSettings.DEFAULTS.precomputeTransitiveLoads(),
            noImplicitFileExport,
            simplifyUnconditionalSelectsInRuleAttrs,
            /* workspaceName= */ DUMMY_WORKSPACE_NAME_FOR_BZLMOD_PACKAGES,
            /* mainRepositoryMapping= */ null,
            /* cpuBoundSemaphore= */ null,
            PackageOverheadEstimator.NOOP_ESTIMATOR,
            /* generatorMap= */ null,
            /* globber= */ null,
            /* enableNameConflictChecking= */ true,
            /* trackFullMacroInformation= */ true,
            PackageLimits.DEFAULTS)
        .setLoads(ImmutableList.of());
  }

  // ==== Non-trivial nested classes ====

  /**
   * Common base class for builders for {@link Package} and {@link PackagePiece.ForBuildFile}
   * objects, containing the shared logic for processing top-level BUILD file declarations, for
   * example the "package" callable.
   */
  // TODO(https://github.com/bazelbuild/bazel/issues/23852): this class should be moved elsewhere -
  // probably to an inner clas of Packageoid - but that would require also moving Declarations and
  // PackageArgs, so that their private fields can be mutated only by the builder.
  public abstract static class AbstractBuilder extends TargetDefinitionContext {
    private final boolean precomputeTransitiveLoads;

    /** True iff the "package" function has already been called in this BUILD file. */
    private boolean packageFunctionUsed;

    protected final boolean noImplicitFileExport;

    /** Retrieves this object from a Starlark thread. Returns null if not present. */
    @Nullable
    public static AbstractBuilder fromOrNull(StarlarkThread thread) {
      StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
      return ctx instanceof AbstractBuilder builder ? builder : null;
    }

    /**
     * Retrieves this object from a Starlark thread. If not present, throws an {@link EvalException}
     * with an error message indicating that {@code what} can only be used in a BUILD file or a
     * legacy macro.
     */
    @CanIgnoreReturnValue
    public static AbstractBuilder fromOrFailAllowBuildOnly(
        StarlarkThread thread, String what, String participle) throws EvalException {
      @Nullable StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
      if (ctx instanceof AbstractBuilder builder
          && builder.recorder.getCurrentMacroFrame() == null) {
        return builder;
      }
      throw newFromOrFailException(
          what,
          participle,
          thread.getSemantics(),
          EnumSet.of(FromOrFailMode.NO_MACROS, FromOrFailMode.NO_WORKSPACE));
    }

    /**
     * Retrieves this object from a Starlark thread. If not present, throws an {@link EvalException}
     * with an error message indicating that {@code what} can only be used in a BUILD file or a
     * legacy macro.
     */
    @CanIgnoreReturnValue
    public static AbstractBuilder fromOrFailAllowBuildOnly(StarlarkThread thread, String what)
        throws EvalException {
      return fromOrFailAllowBuildOnly(thread, what, "used");
    }

    // TODO(#19922): Require this to be set before BUILD evaluation.
    @CanIgnoreReturnValue
    public AbstractBuilder setLoads(Iterable<Module> directLoads) {
      checkLoadsNotSet();
      if (precomputeTransitiveLoads) {
        pkg.getDeclarations().transitiveLoads = computeTransitiveLoads(directLoads);
      } else {
        pkg.getDeclarations().directLoads = ImmutableList.copyOf(directLoads);
      }
      return this;
    }

    static ImmutableList<Label> computeTransitiveLoads(Iterable<Module> directLoads) {
      Set<Label> loads = new LinkedHashSet<>();
      BazelModuleContext.visitLoadGraphRecursively(directLoads, loads::add);
      return ImmutableList.copyOf(loads);
    }

    @CanIgnoreReturnValue
    AbstractBuilder setTransitiveLoadsForDeserialization(ImmutableList<Label> transitiveLoads) {
      checkLoadsNotSet();
      pkg.getDeclarations().transitiveLoads = checkNotNull(transitiveLoads);
      return this;
    }

    private void checkLoadsNotSet() {
      checkState(
          pkg.getDeclarations().directLoads == null,
          "Direct loads already set: %s",
          pkg.getDeclarations().directLoads);
      checkState(
          pkg.getDeclarations().transitiveLoads == null,
          "Transitive loads already set: %s",
          pkg.getDeclarations().transitiveLoads);
    }

    public void mergePackageArgsFrom(PackageArgs packageArgs) {
      pkg.getDeclarations()
          .setPackageArgs(pkg.getDeclarations().getPackageArgs().mergeWith(packageArgs));
    }

    public void mergePackageArgsFrom(PackageArgs.Builder builder) {
      mergePackageArgsFrom(builder.build());
    }

    public void setMakeVariable(String name, String value) {
      makeEnv.put(name, value);
    }

    /** Returns whether the "package" function has been called yet */
    boolean isPackageFunctionUsed() {
      return packageFunctionUsed;
    }

    void setPackageFunctionUsed() {
      packageFunctionUsed = true;
    }

    public Set<Target> getTargets() {
      return recorder.getTargets();
    }

    /** Adds an environment group to the package. Not valid within symbolic macros. */
    void addEnvironmentGroup(
        String name,
        List<Label> environments,
        List<Label> defaults,
        EventHandler eventHandler,
        Location location)
        throws NameConflictException, LabelSyntaxException {
      Preconditions.checkState(currentMacro() == null);

      if (hasDuplicateLabels(environments, name, "environments", location, eventHandler)
          || hasDuplicateLabels(defaults, name, "defaults", location, eventHandler)) {
        setContainsErrors();
        return;
      }

      EnvironmentGroup group =
          new EnvironmentGroup(createLabel(name), pkg, environments, defaults, location);
      recorder.addTarget(group);

      // Invariant: once group is inserted into targets, it must also:
      // (a) be inserted into environmentGroups, or
      // (b) have its group.processMemberEnvironments called.
      // Otherwise it will remain uninitialized,
      // causing crashes when it is later toString-ed.

      for (Event error : group.validateMembership()) {
        eventHandler.handle(error);
        setContainsErrors();
      }

      // For each declared environment, make sure it doesn't also belong to some other group.
      for (Label environment : group.getEnvironments()) {
        EnvironmentGroup otherGroup = environmentGroups.get(environment);
        if (otherGroup != null) {
          eventHandler.handle(
              Package.error(
                  location,
                  String.format(
                      "environment %s belongs to both %s and %s",
                      environment, group.getLabel(), otherGroup.getLabel()),
                  Code.ENVIRONMENT_IN_MULTIPLE_GROUPS));
          setContainsErrors();
          // Ensure the orphan gets (trivially) initialized.
          group.processMemberEnvironments(ImmutableMap.of());
        } else {
          environmentGroups.put(environment, group);
        }
      }
    }

    /**
     * Returns true if any labels in the given list appear multiple times, reporting an appropriate
     * error message if so.
     *
     * <p>TODO(bazel-team): apply this to all build functions (maybe automatically?), possibly
     * integrate with RuleClass.checkForDuplicateLabels.
     */
    private static boolean hasDuplicateLabels(
        List<Label> labels,
        String owner,
        String attrName,
        Location location,
        EventHandler eventHandler) {
      Set<Label> dupes = CollectionUtils.duplicatedElementsOf(labels);
      for (Label dupe : dupes) {
        eventHandler.handle(
            Package.error(
                location,
                String.format(
                    "label '%s' is duplicated in the '%s' list of '%s'", dupe, attrName, owner),
                Code.DUPLICATE_LABEL));
      }
      return !dupes.isEmpty();
    }

    protected void beforeBuildWithoutDiscoveringAssumedInputFiles() throws NoSuchPackageException {
      // We create an InputFile corresponding to the BUILD file in Builder's constructor. However,
      // the visibility of this target may be overridden with an exports_files directive, so we wait
      // until now to obtain the current instance from the targets map.
      setBuildFile((InputFile) recorder.getTargetMap().get(metadata.buildFileLabel().getName()));

      super.beforeBuild();
    }

    protected abstract void setBuildFile(InputFile buildFile);

    @CanIgnoreReturnValue
    @Override
    protected AbstractBuilder beforeBuild() throws NoSuchPackageException {
      beforeBuildWithoutDiscoveringAssumedInputFiles();
      Map<String, InputFile> newInputFiles =
          createAssumedInputFiles(pkg, recorder, noImplicitFileExport);
      for (InputFile file : newInputFiles.values()) {
        recorder.addInputFileUnchecked(file);
      }
      return this;
    }

    /**
     * Creates and returns input files for targets that have been referenced but not explicitly
     * declared in this package.
     *
     * <p>Precisely: For each label L appearing in one or more label-typed attributes of one or more
     * declarations D (either of a target or a symbolic macro), we create an {@code InputFile} for L
     * and return it in the map (keyed by its name) if all of the following are true:
     *
     * <ol>
     *   <li>L points to within the current package.
     *   <li>The package does not otherwise declare a target or macro named L.
     *   <li>D is not itself declared inside a symbolic macro.
     * </ol>
     *
     * <p>The third condition ensures that we can know all *possible* implicitly created input files
     * without evaluating any symbolic macros. However, if the label lies within one or more
     * symbolic macro's namespaces, then we do still need to evaluate those macros to determine
     * whether or not the second condition is true, i.e. whether the label points to a target the
     * macro declares (or a submacro it clashes with), or defaults to an implicitly created input
     * file.
     */
    private static Map<String, InputFile> createAssumedInputFiles(
        Packageoid pkg, TargetRecorder recorder, boolean noImplicitFileExport) {
      Map<String, InputFile> implicitlyCreatedInputFiles = new HashMap<>();

      for (Rule rule : recorder.getRules()) {
        if (!recorder.isRuleCreatedInMacro(rule)) {
          for (Label label : recorder.getRuleLabels(rule)) {
            maybeCreateAssumedInputFile(
                implicitlyCreatedInputFiles,
                pkg,
                recorder,
                noImplicitFileExport,
                label,
                rule.getLocation());
          }
        }
      }

      for (MacroInstance macro : recorder.getMacroMap().values()) {
        if (macro.getParent() == null) {
          macro.visitExplicitAttributeLabels(
              label ->
                  maybeCreateAssumedInputFile(
                      implicitlyCreatedInputFiles,
                      pkg,
                      recorder,
                      noImplicitFileExport,
                      label,
                      // TODO(bazel-team): We don't save a MacroInstance's location information yet,
                      // but when we do, use that here.
                      Location.BUILTIN));
        }
      }

      return implicitlyCreatedInputFiles;
    }

    /**
     * Adds an implicitly created input file to the given map if the label points within the current
     * package and there is no existing target or macro for that label.
     */
    private static void maybeCreateAssumedInputFile(
        Map<String, InputFile> implicitlyCreatedInputFiles,
        Packageoid pkg,
        TargetRecorder recorder,
        boolean noImplicitFileExport,
        Label label,
        Location loc) {
      String name = label.getName();
      if (!label.getPackageIdentifier().equals(pkg.getPackageIdentifier())) {
        return;
      }
      if (recorder.getTargetMap().containsKey(name)
          || recorder.hasMacroWithName(name)
          || implicitlyCreatedInputFiles.containsKey(name)) {
        return;
      }

      implicitlyCreatedInputFiles.put(
          name,
          noImplicitFileExport
              ? new PrivateVisibilityInputFile(pkg, label, loc)
              : new InputFile(pkg, label, loc));
    }

    @Override
    protected void finalBuilderValidationHook() {
      // Now all targets have been loaded, so we validate the group's member environments.
      for (EnvironmentGroup envGroup : ImmutableSet.copyOf(environmentGroups.values())) {
        List<Event> errors = envGroup.processMemberEnvironments(recorder.getTargetMap());
        if (!errors.isEmpty()) {
          Event.replayEventsOn(localEventHandler, errors);
          // TODO(bazel-team): Can't we automatically infer containsError from the presence of
          // ERRORs on our handler?
          setContainsErrors();
        }
      }
    }

    @Override
    protected void packageoidInitializationHook() {
      // Finish Package.Declarations construction.
      if (pkg.getDeclarations().directLoads == null
          && pkg.getDeclarations().transitiveLoads == null) {
        checkState(pkg.containsErrors(), "Loads not set for error-free package");
        setLoads(ImmutableList.of());
      }
      pkg.getDeclarations().workspaceName = workspaceName;
      pkg.getDeclarations().makeEnv = ImmutableMap.copyOf(makeEnv);
    }

    AbstractBuilder(
        Package.Metadata metadata,
        Packageoid pkg,
        SymbolGenerator<?> symbolGenerator,
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
        boolean enableTargetMapSnapshotting,
        PackageLimits packageLimits) {
      super(
          metadata,
          pkg,
          symbolGenerator,
          simplifyUnconditionalSelectsInRuleAttrs,
          workspaceName,
          mainRepositoryMapping,
          cpuBoundSemaphore,
          packageOverheadEstimator,
          generatorMap,
          globber,
          enableNameConflictChecking,
          trackFullMacroInformation,
          enableTargetMapSnapshotting,
          packageLimits);
      this.precomputeTransitiveLoads = precomputeTransitiveLoads;
      this.noImplicitFileExport = noImplicitFileExport;
      if (metadata.getName().startsWith("javatests/")) {
        mergePackageArgsFrom(PackageArgs.builder().setDefaultTestOnly(true));
      }
      // Add target for the BUILD file itself.
      // (This may be overridden by an exports_file declaration.)
      recorder.addInputFileUnchecked(
          new InputFile(pkg, metadata.buildFileLabel(), metadata.getBuildFileLocation()));
    }
  }

  /**
   * A builder for {@link Package} objects. Only intended to be used by {@link PackageFactory} and
   * {@link com.google.devtools.build.lib.skyframe.PackageFunction}.
   */
  public static class Builder extends AbstractBuilder {

    /**
     * A bundle of options affecting package construction, that is not specific to any particular
     * package.
     */
    public interface PackageSettings {
      /**
       * Returns whether or not extra detail should be added to {@link NoSuchTargetException}s
       * thrown from {@link #getTarget}. Useful for toning down verbosity in situations where it can
       * be less helpful.
       */
      // TODO(bazel-team): Arguably, this could be replaced by a boolean param to getTarget(), or
      // some separate action taken by the caller. But there's a lot of call sites that would need
      // updating.
      default boolean succinctTargetNotFoundErrors() {
        return false;
      }

      /**
       * Determines whether to precompute a list of transitively loaded starlark files while
       * building packages.
       *
       * <p>Typically, direct loads are stored as a {@code ImmutableList<Module>}. This is
       * sufficient to reconstruct the full load graph by recursively traversing {@link
       * BazelModuleContext#loads}. If the package is going to be serialized, however, it may make
       * more sense to precompute a flat list containing the labels of all transitively loaded bzl
       * files since {@link Module} is costly to serialize.
       *
       * <p>If this returns {@code true}, transitive loads are stored as an {@code
       * ImmutableList<Label>} and direct loads are not stored.
       */
      default boolean precomputeTransitiveLoads() {
        return false;
      }

      PackageSettings DEFAULTS = new PackageSettings() {};
    }

    /** A bundle of options affecting resource limits on package construction. */
    public interface PackageLimits {
      /**
       * The maximum number of Starlark computation steps that are allowed to be executed while
       * building a package (or, transitively, any package piece). If this limit is exceeded, the
       * package or package piece immediately stops building.
       *
       * <p>Confusingly, for historical Google-specific reasons, this limit is <em>not</em> the same
       * as {@code --max_computation_steps}.
       *
       * <ul>
       *   <li>This limit (maxStarlarkComputationStepsPerPackage) is only set by Google-specific
       *       logic, is currently not used in open-source Bazel, and exceeding the limit causes the
       *       package builder to immediately stop and print a stack trace. The intent is to harden
       *       infrastructure against runaway Starlark computations.
       *   <li>By contrast, {@code --max_computation_steps} is enforced by {@link PackageFactory}
       *       post-factum, after the package has been built. The intent is to enforce code health
       *       by limiting the complexity of packages in a repo.
       * </ul>
       *
       * <p>If lazy symbolic macro expansion is enabled, unless a complete {@link Package} is
       * loaded, the limit is enforced only per package piece.
       */
      // TODO(b/417468797): merge with --max_computation_steps enforcement.
      default long maxStarlarkComputationStepsPerPackage() {
        return Long.MAX_VALUE;
      }

      public static final PackageLimits DEFAULTS = new PackageLimits() {};
    }

    // The snapshot of {@link #targets} for use in rule finalizer macros. Contains all
    // non-finalizer-instantiated rule targets (i.e. all rule targets except for those instantiated
    // in a finalizer or in a macro called from a finalizer).
    //
    // Initialized by expandAllRemainingMacros() and reset to null by beforeBuild().
    @Nullable private Map<String, Rule> rulesSnapshotViewForFinalizers;

    /**
     * Ids of all symbolic macros that have been declared but not yet evaluated.
     *
     * <p>These are listed in the order they were declared. (This probably doesn't matter, but let's
     * be protective against possible non-determinism.)
     *
     * <p>Generally, ordinary symbolic macros are evaluated eagerly and not added to this set, while
     * finalizers, as well as any macros called by finalizers, always use deferred evaluation and
     * end up in here.
     */
    private final Set<String> unexpandedMacros = new LinkedHashSet<>();

    private Builder(
        Metadata metadata,
        SymbolGenerator<?> symbolGenerator,
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
          metadata,
          new Package(metadata),
          symbolGenerator,
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
          /* enableTargetMapSnapshotting= */ true,
          packageLimits);
    }

    /** Retrieves this object from a Starlark thread. Returns null if not present. */
    @Nullable
    public static Builder fromOrNull(StarlarkThread thread) {
      StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
      return ctx instanceof Builder builder ? builder : null;
    }

    /**
     * Retrieves this object from a Starlark thread. If not present, or if this object is not a repo
     * rule package builder, throws an {@link EvalException} with an error message indicating that
     * {@code what} can only be used in a module extension.
     */
    @CanIgnoreReturnValue
    public static Builder fromOrFailAllowModuleExtension(
        StarlarkThread thread, String what, String participle) throws EvalException {
      @Nullable StarlarkThreadContext ctx = thread.getThreadLocal(StarlarkThreadContext.class);
      if (ctx instanceof Package.Builder pkgBuilder && pkgBuilder.isRepoRulePackage()) {
        return pkgBuilder;
      }
      throw Starlark.errorf(
          "%s can only be %s while evaluating the implementation function of a module extension",
          what, participle);
    }

    Package getPackage() {
      return (Package) pkg;
    }

    @Override
    @CanIgnoreReturnValue
    public Builder setLoads(Iterable<Module> directLoads) {
      return (Builder) super.setLoads(directLoads);
    }

    @Override
    Map<String, Rule> getRulesSnapshotView() {
      if (rulesSnapshotViewForFinalizers != null) {
        return rulesSnapshotViewForFinalizers;
      } else {
        return super.getRulesSnapshotView();
      }
    }

    @Override
    @Nullable
    Rule getNonFinalizerInstantiatedRule(String name) {
      if (rulesSnapshotViewForFinalizers != null) {
        return rulesSnapshotViewForFinalizers.get(name);
      } else {
        return super.getNonFinalizerInstantiatedRule(name);
      }
    }

    public void addRuleUnchecked(Rule rule) {
      Preconditions.checkArgument(rule.getPackage() == pkg);
      recorder.addRuleUnchecked(rule);
    }

    @Override
    public boolean eagerlyExpandMacros() {
      return true;
    }

    @Override
    public void addMacro(MacroInstance macro) throws NameConflictException {
      super.addMacro(macro);
      unexpandedMacros.add(macro.getId());
    }

    // For Package deserialization.
    void putAllMacroNamespaceViolatingTargets(Map<String, String> macroNamespaceViolatingTargets) {
      recorder.putAllMacroNamespaceViolatingTargets(macroNamespaceViolatingTargets);
    }

    void putAllTargetsToDeclaringPackage(Map<String, PackageIdentifier> targetsToDeclaringPackage) {
      recorder.putAllTargetsToDeclaringPackage(targetsToDeclaringPackage);
    }

    /**
     * Marks a symbolic macro as having finished evaluating.
     *
     * <p>This will prevent the macro from being run by {@link #expandAllRemainingMacros}.
     *
     * <p>The macro must not have previously been marked complete.
     */
    public void markMacroComplete(MacroInstance macro) {
      String id = macro.getId();
      if (!unexpandedMacros.remove(id)) {
        throw new IllegalArgumentException(
            String.format("Macro id '%s' unknown or already marked complete", id));
      }
    }

    /**
     * Ensures that all symbolic macros in an error-free package have expanded. No-op if the package
     * already {@link #containsErrors}.
     *
     * <p>This does not run any macro that has already been evaluated. It *does* run macros that are
     * newly discovered during the operation of this method.
     */
    public void expandAllRemainingMacros(StarlarkSemantics semantics)
        throws EvalException, InterruptedException {
      // TODO: #19922 - Protect against unreasonable macro stack depth and large numbers of symbolic
      // macros overall, for both the eager and deferred evaluation strategies.

      // Note that this operation is idempotent for symmetry with build()/buildPartial(). Though
      // it's not entirely clear that this is necessary.

      // TODO: #19922 - Once compatibility with native.existing_rules() in legacy macros is no
      // longer a concern, we will want to support delayed expansion of non-finalizer macros before
      // the finalizer expansion step.

      // Finalizer expansion step. Requires that the package not be in error (no point in finalizing
      // a package that already threw an EvalException).
      if (!containsErrors() && !unexpandedMacros.isEmpty()) {
        Preconditions.checkState(
            unexpandedMacros.stream()
                .allMatch(id -> recorder.getMacroMap().get(id).getMacroClass().isFinalizer()),
            "At the beginning of finalizer expansion, unexpandedMacros must contain only"
                + " finalizers");

        // Save a snapshot of rule targets for use by native.existing_rules() inside all finalizers.
        // We must take this snapshot before calling any finalizer because the snapshot must not
        // include any rule instantiated by a finalizer or macro called from a finalizer.
        if (rulesSnapshotViewForFinalizers == null) {
          Preconditions.checkState(
              recorder.getTargetMap() instanceof SnapshottableBiMap<?, ?>,
              "Cannot call expandAllRemainingMacros() after beforeBuild() has been called");
          rulesSnapshotViewForFinalizers = getRulesSnapshotView();
        }

        while (!unexpandedMacros.isEmpty()) { // NB: collection mutated by body
          String id = unexpandedMacros.iterator().next();
          MacroInstance macro = recorder.getMacroMap().get(id);
          MacroClass.executeMacroImplementation(macro, this, semantics);
        }
      }
    }

    @Override
    @CanIgnoreReturnValue
    protected Builder beforeBuild() throws NoSuchPackageException {
      // For correct semantics, we refuse to build a package that hasn't thrown any EvalExceptions
      // but has declared symbolic macros that have not yet been expanded. (Currently finalizers are
      // the only use case where this happens, but the Package logic is agnostic to that detail.)
      //
      // Production code should be calling expandAllRemainingMacros() to guarantee that nothing is
      // left unexpanded. Tests that do not declare any symbolic macros need not make the call.
      // Package deserialization doesn't have to do it either, since we shouldn't be evaluating
      // symbolic macros on the deserialized result of an already evaluated package.
      Preconditions.checkState(
          unexpandedMacros.isEmpty() || containsErrors(),
          "Cannot build a package with unexpanded symbolic macros; call"
              + " expandAllRemainingMacros()");

      // SnapshottableBiMap does not allow removing targets (in order to efficiently track rule
      // insertion order). However, we *do* need to support removal of targets in
      // PackageFunction.handleLabelsCrossingSubpackagesAndPropagateInconsistentFilesystemExceptions
      // which is called *between* calls to beforeBuild and finishBuild. We thus repoint the targets
      // map to the SnapshottableBiMap's underlying bimap and thus stop tracking insertion order.
      // After this point, snapshots of targets should no longer be used, and any further
      // getRulesSnapshotView calls will throw.
      if (recorder.getTargetMap() instanceof SnapshottableBiMap<?, ?>) {
        recorder.unwrapSnapshottableBiMap();
        rulesSnapshotViewForFinalizers = null;
      }

      super.beforeBuild();

      return this;
    }

    @Override
    @CanIgnoreReturnValue
    public Builder buildPartial() throws NoSuchPackageException {
      return (Builder) super.buildPartial();
    }

    @Override
    protected void setBuildFile(InputFile buildFile) {
      ((Package) pkg).buildFile = checkNotNull(buildFile);
    }

    @Override
    public Package finishBuild() {
      return (Package) super.finishBuild();
    }

    @Override
    protected void packageoidInitializationHook() {
      super.packageoidInitializationHook();
      Package pkg = getPackage();
      pkg.computationSteps = getComputationSteps();
      pkg.macros = ImmutableSortedMap.copyOf(recorder.getMacroMap());
      pkg.macroNamespaceViolatingTargets =
          ImmutableMap.copyOf(recorder.getMacroNamespaceViolatingTargets());
      pkg.targetsToDeclaringMacro =
          recorder.getTargetsToDeclaringMacro() != null
              ? ImmutableSortedMap.copyOf(recorder.getTargetsToDeclaringMacro())
              : null;
      pkg.targetsToDeclaringPackage =
          recorder.getTargetsToDeclaringPackage() != null
              ? ImmutableSortedMap.copyOf(recorder.getTargetsToDeclaringPackage())
              : null;
    }

    /** Completes package construction. Idempotent. */
    // TODO(brandjon): Do we actually care about idempotence?
    public Package build() throws NoSuchPackageException {
      return build(/* discoverAssumedInputFiles= */ true);
    }

    /**
     * Constructs the package (or does nothing if it's already built) and returns it.
     *
     * @param discoverAssumedInputFiles whether to automatically add input file targets to this
     *     package for "dangling labels", i.e. labels mentioned in this package that point to an
     *     up-until-now non-existent target in this package
     */
    Package build(boolean discoverAssumedInputFiles) throws NoSuchPackageException {
      if (alreadyBuilt) {
        return getPackage();
      }
      if (discoverAssumedInputFiles) {
        beforeBuild();
      } else {
        beforeBuildWithoutDiscoveringAssumedInputFiles();
      }
      return finishBuild();
    }
  }

  /** A collection of data that is known before BUILD file evaluation even begins. */
  // TODO(bazel-team): move to Packageoid.java or to its own file to reduce size of Package.java?
  @AutoCodec
  public record Metadata(
      PackageIdentifier packageIdentifier,
      RootedPath buildFilename,
      Label buildFileLabel,
      boolean isRepoRulePackage,
      RepositoryMapping repositoryMapping,
      Optional<String> associatedModuleName,
      Optional<String> associatedModuleVersion,
      @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy,
      boolean succinctTargetNotFoundErrors,
      Optional<Root> sourceRoot) {

    public static Builder builder() {
      return new AutoBuilder_Package_Metadata_Builder();
    }

    /** Builder for {@link Metadata}. */
    @AutoBuilder(callMethod = "of")
    public interface Builder {
      Builder packageIdentifier(PackageIdentifier packageIdentifier);

      Builder buildFilename(RootedPath buildFilename);

      Builder isRepoRulePackage(boolean isRepoRulePackage);

      Builder repositoryMapping(RepositoryMapping repositoryMapping);

      Builder associatedModuleName(Optional<String> associatedModuleName);

      Builder associatedModuleVersion(Optional<String> associatedModuleVersion);

      Builder configSettingVisibilityPolicy(
          @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy);

      Builder succinctTargetNotFoundErrors(boolean succinctTargetNotFoundErrors);

      Metadata build();
    }

    static Metadata of(
        PackageIdentifier packageIdentifier,
        RootedPath buildFilename,
        boolean isRepoRulePackage,
        RepositoryMapping repositoryMapping,
        Optional<String> associatedModuleName,
        Optional<String> associatedModuleVersion,
        @Nullable ConfigSettingVisibilityPolicy configSettingVisibilityPolicy,
        boolean succinctTargetNotFoundErrors) {
      Label buildFileLabel;
      try {
        buildFileLabel =
            Label.create(packageIdentifier, buildFilename.getRootRelativePath().getBaseName());
      } catch (LabelSyntaxException e) {
        // This can't actually happen.
        throw new AssertionError("Package BUILD file has an illegal name: " + buildFilename, e);
      }
      return new Metadata(
          packageIdentifier,
          buildFilename,
          buildFileLabel,
          isRepoRulePackage,
          repositoryMapping,
          associatedModuleName,
          associatedModuleVersion,
          configSettingVisibilityPolicy,
          succinctTargetNotFoundErrors,
          computeSourceRoot(packageIdentifier, buildFilename, isRepoRulePackage));
    }

    /**
     * @deprecated Use {@link #builder()} instead.
     */
    @Deprecated
    public Metadata {
      Preconditions.checkNotNull(packageIdentifier);
      Preconditions.checkNotNull(buildFilename);
      Preconditions.checkNotNull(repositoryMapping);
      Preconditions.checkNotNull(associatedModuleName);
      Preconditions.checkNotNull(associatedModuleVersion);
      Preconditions.checkNotNull(sourceRoot);

      // Check for consistency between isRepoRulePackage and whether the buildFilename is a
      // MODULE.bazel file.
      String baseName = buildFilename.asPath().getBaseName();
      boolean isModuleDotBazelFile =
          baseName.equals(LabelConstants.MODULE_DOT_BAZEL_FILE_NAME.getPathString());
      Preconditions.checkArgument(isRepoRulePackage == isModuleDotBazelFile);
    }

    /** Returns the name of this package (sans repository), e.g. "foo/bar". */
    public String getName() {
      return packageIdentifier.getPackageFragment().getPathString();
    }

    /**
     * Returns the directory in which this package's BUILD file resides.
     *
     * <p>All InputFile members of the packages are located relative to this directory.
     */
    public Path getPackageDirectory() {
      return getPackageDirectory(buildFilename);
    }

    /** Returns the {@link Location} of the package's BUILD file. */
    public Location getBuildFileLocation() {
      return Location.fromFile(buildFilename.asPath().toString());
    }

    private static Path getPackageDirectory(RootedPath buildFilename) {
      return buildFilename.asPath().getParentDirectory();
    }

    private static Optional<Root> computeSourceRoot(
        PackageIdentifier packageIdentifier, RootedPath buildFilename, boolean isRepoRulePackage) {
      Preconditions.checkNotNull(packageIdentifier);
      Preconditions.checkNotNull(buildFilename);
      if (isRepoRulePackage) {
        return Optional.empty();
      }

      RootedPath buildFileRootedPath = buildFilename;
      Root buildFileRoot = buildFileRootedPath.getRoot();
      PathFragment pkgIdFragment = packageIdentifier.getSourceRoot();
      PathFragment pkgDirFragment = buildFileRootedPath.getRootRelativePath().getParentDirectory();

      Root sourceRoot;
      if (pkgIdFragment.equals(pkgDirFragment)) {
        // Fast path: BUILD file path and package name are the same, don't create an extra root.
        sourceRoot = buildFileRoot;
      } else {
        // TODO(bazel-team): Can this expr be simplified to just pkgDirFragment?
        PathFragment current = buildFileRootedPath.asPath().asFragment().getParentDirectory();
        for (int i = 0, len = pkgIdFragment.segmentCount(); i < len && current != null; i++) {
          current = current.getParentDirectory();
        }
        if (current == null || current.isEmpty()) {
          // This is never really expected to work. The below check should fail.
          sourceRoot = buildFileRoot;
        } else {
          // Note that current is an absolute path.
          sourceRoot = Root.fromPath(buildFileRoot.getRelative(current));
        }
      }

      Preconditions.checkArgument(
          sourceRoot.asPath() != null
              && sourceRoot.getRelative(pkgIdFragment).equals(getPackageDirectory(buildFilename)),
          "Invalid BUILD file name for package '%s': %s (in source %s with packageDirectory %s and"
              + " package identifier source root %s)",
          packageIdentifier,
          buildFilename,
          sourceRoot,
          getPackageDirectory(buildFilename),
          packageIdentifier.getSourceRoot());

      return Optional.of(sourceRoot);
    }
  }

  /**
   * A collection of data about a package that is known after BUILD file evaluation has completed,
   * which doesn't require expanding any symbolic macros, and which transitively doesn't hold
   * references to {@link Package} or {@link PackagePiece} objects. Treated as immutable after BUILD
   * file evaluation has completed.
   *
   * <p>This class should not be extended - it's only non-final for mocking!
   */
  public static class Declarations {
    // For BUILD files, this is initialized immediately. For WORKSPACE files, it is known only after
    // Starlark evaluation of the WORKSPACE file has finished.
    // TODO(bazel-team): move to Metadata when WORKSPACE file logic is deleted.
    private String workspaceName;

    // Mutated during BUILD file evaluation (but not by symbolic macro evaluation).
    private PackageArgs packageArgs = PackageArgs.DEFAULT;

    // Mutated during BUILD file evaluation (but not by symbolic macro evaluation).
    private ImmutableMap<String, String> makeEnv;

    // These two fields are mutually exclusive. Which one is set depends on
    // PackageSettings#precomputeTransitiveLoads. See Package.Builder#setLoads.
    @Nullable private ImmutableList<Module> directLoads;
    @Nullable private ImmutableList<Label> transitiveLoads;

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      return obj instanceof Declarations other
          && Objects.equals(workspaceName, other.workspaceName)
          && Objects.equals(packageArgs, other.packageArgs)
          && Objects.equals(makeEnv, other.makeEnv)
          // Serializers use getOrComputeTransitivelyLoadedStarlarkFiles() and don't distinguish
          // between directLoads and transitiveLoads.
          && Objects.equals(
              getOrComputeTransitivelyLoadedStarlarkFilesInternal(),
              other.getOrComputeTransitivelyLoadedStarlarkFilesInternal());
    }

    @Override
    public int hashCode() {
      return HashCodes.hashObjects(
          workspaceName,
          packageArgs,
          makeEnv,
          // Serializers use getOrComputeTransitivelyLoadedStarlarkFiles() and don't distinguish
          // between directLoads and transitiveLoads.
          getOrComputeTransitivelyLoadedStarlarkFilesInternal());
    }

    /**
     * Returns the name of the workspace this package is in. Used as a prefix for the runfiles
     * directory.
     */
    public String getWorkspaceName() {
      return workspaceName;
    }

    /**
     * Returns the collection of package-level attributes set by the {@code package()} callable and
     * similar methods.
     */
    public PackageArgs getPackageArgs() {
      return packageArgs;
    }

    /**
     * Sets the arguments of the {@code package()} callable.
     *
     * <p>This method should only be used by builders for {@link Package} and {@link
     * PackagePiece.ForBuildFile}.
     */
    void setPackageArgs(PackageArgs packageArgs) {
      this.packageArgs = packageArgs;
    }

    /**
     * Returns the "Make" environment of this package, containing package-local definitions of
     * "Make" variables.
     */
    public ImmutableMap<String, String> getMakeEnvironment() {
      return makeEnv;
    }

    /**
     * Returns a list of Starlark files transitively loaded by this package.
     *
     * <p>If transitive loads are not {@linkplain PackageSettings#precomputeTransitiveLoads
     * precomputed}, performs a traversal over the load graph to compute them.
     *
     * <p>If only the count of transitively loaded files is needed, use {@link
     * #countTransitivelyLoadedStarlarkFiles}. For a customized online visitation, use {@link
     * #visitLoadGraph}.
     *
     * <p>This method can only be used after the Package or PackagePiece has been fully initialized
     * (i.e. after {@link TargetDefinitionContext#finishBuild} has been called).
     */
    public ImmutableList<Label> getOrComputeTransitivelyLoadedStarlarkFiles() {
      return checkNotNull(getOrComputeTransitivelyLoadedStarlarkFilesInternal());
    }

    @Nullable
    private ImmutableList<Label> getOrComputeTransitivelyLoadedStarlarkFilesInternal() {
      if (transitiveLoads != null) {
        return transitiveLoads;
      } else if (directLoads != null) {
        return computeTransitiveLoads(directLoads);
      } else {
        // Declarations not fully initialized.
        return null;
      }
    }

    /**
     * Counts the number Starlark files transitively loaded by this package.
     *
     * <p>If transitive loads are not {@linkplain PackageSettings#precomputeTransitiveLoads
     * precomputed}, performs a traversal over the load graph to count them.
     *
     * <p>This method can only be used after the Package or PackagePiece has been fully initialized
     * (i.e. after {@link TargetDefinitionContext#finishBuild} has been called).
     */
    public int countTransitivelyLoadedStarlarkFiles() {
      if (transitiveLoads != null) {
        return transitiveLoads.size();
      }
      Set<Label> loads = new HashSet<>();
      visitLoadGraph(loads::add);
      return loads.size();
    }

    /**
     * Performs an online visitation of the load graph rooted at this package.
     *
     * <p>If transitive loads were {@linkplain PackageSettings#precomputeTransitiveLoads
     * precomputed}, each file is passed to {@link LoadGraphVisitor#visit} once regardless of its
     * return value.
     *
     * <p>This method can only be used after the Package or PackagePiece has been fully initialized
     * (i.e. after {@link TargetDefinitionContext#finishBuild} has been called).
     */
    public <E1 extends Exception, E2 extends Exception> void visitLoadGraph(
        LoadGraphVisitor<E1, E2> visitor) throws E1, E2 {
      if (transitiveLoads != null) {
        for (Label load : transitiveLoads) {
          visitor.visit(load);
        }
      } else {
        BazelModuleContext.visitLoadGraphRecursively(directLoads, visitor);
      }
    }

    /**
     * Objects of this class should only be constructed by constructors for {@link Package} and
     * {@link PackagePiece.ForBuildFile}.
     */
    Declarations() {}
  }

  /** Package codec implementation. */
  @VisibleForTesting
  static final class PackageCodec implements ObjectCodec<Package> {
    @Override
    public Class<Package> getEncodedClass() {
      return Package.class;
    }

    @Override
    public void serialize(SerializationContext context, Package input, CodedOutputStream codedOut)
        throws IOException, SerializationException {
      context.checkClassExplicitlyAllowed(Package.class, input);
      PackageCodecDependencies codecDeps = context.getDependency(PackageCodecDependencies.class);
      codecDeps.getPackageSerializer().serialize(context, input, codedOut);
    }

    @Override
    public Package deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      PackageCodecDependencies codecDeps = context.getDependency(PackageCodecDependencies.class);
      return codecDeps.getPackageSerializer().deserialize(context, codedIn);
    }
  }
}
