// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;

/** Provides a way to access attributes that are common to all compilation rules. */
// TODO(bazel-team): Delete and move into support-specific attributes classes once ObjcCommon is
// gone.
final class CompilationAttributes implements StarlarkValue {
  static class Builder {
    private final NestedSetBuilder<Artifact> hdrs = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> textualHdrs = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<PathFragment> includes = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<PathFragment> sdkIncludes = NestedSetBuilder.stableOrder();
    private final ImmutableList.Builder<String> copts = ImmutableList.builder();
    private final ImmutableList.Builder<String> linkopts = ImmutableList.builder();
    private final ImmutableList.Builder<Artifact> linkInputs = ImmutableList.builder();
    private final ImmutableList.Builder<String> defines = ImmutableList.builder();
    private final NestedSetBuilder<String> sdkFrameworks = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<String> weakSdkFrameworks = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<String> sdkDylibs = NestedSetBuilder.stableOrder();
    private Optional<PathFragment> packageFragment = Optional.absent();
    private boolean enableModules;

    /** Adds the default values available through the rule's context. */
    static Builder fromRuleContext(RuleContext ruleContext) throws InterruptedException {
      Builder builder = new Builder();

      addHeadersFromRuleContext(builder, ruleContext);
      addIncludesFromRuleContext(builder, ruleContext);
      addSdkAttributesFromRuleContext(builder, ruleContext);
      addCompileOptionsFromRuleContext(builder, ruleContext);
      addModuleOptionsFromRuleContext(builder, ruleContext);

      return builder;
    }

    /** Adds headers to be made available for dependents. */
    @CanIgnoreReturnValue
    public Builder addHdrs(NestedSet<Artifact> hdrs) {
      this.hdrs.addTransitive(hdrs);
      return this;
    }

    /** Adds headers that cannot be compiled individually. */
    @CanIgnoreReturnValue
    public Builder addTextualHdrs(NestedSet<Artifact> textualHdrs) {
      this.textualHdrs.addTransitive(textualHdrs);
      return this;
    }

    /** Adds include paths to be made available for compilation. */
    @CanIgnoreReturnValue
    public Builder addIncludes(NestedSet<PathFragment> includes) {
      this.includes.addTransitive(includes);
      return this;
    }

    /** Adds paths for SDK includes. */
    @CanIgnoreReturnValue
    public Builder addSdkIncludes(NestedSet<PathFragment> sdkIncludes) {
      this.sdkIncludes.addTransitive(sdkIncludes);
      return this;
    }

    /** Adds compile-time options. */
    @CanIgnoreReturnValue
    public Builder addCopts(Iterable<String> copts) {
      this.copts.addAll(copts);
      return this;
    }

    /** Adds link-time options. */
    @CanIgnoreReturnValue
    public Builder addLinkopts(Iterable<String> linkopts) {
      this.linkopts.addAll(linkopts);
      return this;
    }

    /** Adds additional linker inputs. */
    @CanIgnoreReturnValue
    public Builder addLinkInputs(Iterable<Artifact> linkInputs) {
      this.linkInputs.addAll(linkInputs);
      return this;
    }

    /** Adds defines. */
    @CanIgnoreReturnValue
    public Builder addDefines(Iterable<String> defines) {
      this.defines.addAll(defines);
      return this;
    }

    /** Adds SDK frameworks to link against. */
    @CanIgnoreReturnValue
    public Builder addSdkFrameworks(NestedSet<String> sdkFrameworks) {
      this.sdkFrameworks.addTransitive(sdkFrameworks);
      return this;
    }

    /** Adds SDK frameworks to be linked weakly. */
    @CanIgnoreReturnValue
    public Builder addWeakSdkFrameworks(NestedSet<String> weakSdkFrameworks) {
      this.weakSdkFrameworks.addTransitive(weakSdkFrameworks);
      return this;
    }

    /** Adds SDK Dylibs to link against. */
    @CanIgnoreReturnValue
    public Builder addSdkDylibs(NestedSet<String> sdkDylibs) {
      this.sdkDylibs.addTransitive(sdkDylibs);
      return this;
    }

    /** Sets the package path from which to base the header search paths. */
    @CanIgnoreReturnValue
    public Builder setPackageFragment(PathFragment packageFragment) {
      Preconditions.checkState(
          !this.packageFragment.isPresent(),
          "packageFragment is already set to %s",
          this.packageFragment);
      this.packageFragment = Optional.of(packageFragment);
      return this;
    }

    /** Enables the usage of clang modules maps during compilation. */
    @CanIgnoreReturnValue
    public Builder enableModules() {
      this.enableModules = true;
      return this;
    }

    /**
     * Builds a {@code CompilationAttributes} object.
     */
    public CompilationAttributes build() {
      return new CompilationAttributes(
          this.hdrs.build(),
          this.textualHdrs.build(),
          this.includes.build(),
          this.sdkIncludes.build(),
          this.sdkFrameworks.build(),
          this.weakSdkFrameworks.build(),
          this.sdkDylibs.build(),
          this.packageFragment,
          this.copts.build(),
          this.linkopts.build(),
          this.linkInputs.build(),
          this.defines.build(),
          this.enableModules);
    }

    static void addHeadersFromRuleContext(Builder builder, RuleContext ruleContext) {
      if (ruleContext.attributes().has("hdrs", BuildType.LABEL_LIST)) {
        NestedSetBuilder<Artifact> headers = NestedSetBuilder.stableOrder();
        for (Pair<Artifact, Label> header : CcCommon.getHeaders(ruleContext)) {
          headers.add(header.first);
        }
        builder.addHdrs(headers.build());
      }

      if (ruleContext.attributes().has("textual_hdrs", BuildType.LABEL_LIST)) {
        builder.addTextualHdrs(PrerequisiteArtifacts.nestedSet(ruleContext, "textual_hdrs"));
      }
    }

    static void addIncludesFromRuleContext(Builder builder, RuleContext ruleContext) {
      if (ruleContext.attributes().has("includes", Types.STRING_LIST)) {
        NestedSetBuilder<PathFragment> includes = NestedSetBuilder.stableOrder();
        includes.addAll(
            Iterables.transform(
                ruleContext.attributes().get("includes", Types.STRING_LIST), PathFragment::create));
        builder.addIncludes(includes.build());
      }

      if (ruleContext.attributes().has("sdk_includes", Types.STRING_LIST)) {
        NestedSetBuilder<PathFragment> sdkIncludes = NestedSetBuilder.stableOrder();
        sdkIncludes.addAll(
            Iterables.transform(
                ruleContext.attributes().get("sdk_includes", Types.STRING_LIST),
                PathFragment::create));
        builder.addSdkIncludes(sdkIncludes.build());
      }
    }

    static void addSdkAttributesFromRuleContext(Builder builder, RuleContext ruleContext) {
      if (ruleContext.attributes().has("sdk_frameworks", Types.STRING_LIST)) {
        NestedSetBuilder<String> frameworks = NestedSetBuilder.stableOrder();
        for (String explicit : ruleContext.attributes().get("sdk_frameworks", Types.STRING_LIST)) {
          frameworks.add(explicit);
        }
        if (ruleContext.getFragment(ObjcConfiguration.class).disallowSdkFrameworksAttributes()
            && !frameworks.isEmpty()) {
          ruleContext.attributeError(
              "sdk_frameworks",
              "sdk_frameworks attribute is disallowed. Use explicit dependencies instead.");
        }
        builder.addSdkFrameworks(frameworks.build());
      }

      if (ruleContext.attributes().has("weak_sdk_frameworks", Types.STRING_LIST)) {
        NestedSetBuilder<String> weakFrameworks = NestedSetBuilder.stableOrder();
        for (String frameworkName :
            ruleContext.attributes().get("weak_sdk_frameworks", Types.STRING_LIST)) {
          weakFrameworks.add(frameworkName);
        }
        if (ruleContext.getFragment(ObjcConfiguration.class).disallowSdkFrameworksAttributes()
            && !weakFrameworks.isEmpty()) {
          ruleContext.attributeError(
              "weak_sdk_frameworks",
              "weak_sdk_frameworks attribute is disallowed.  Use explicit dependencies instead.");
        }
        builder.addWeakSdkFrameworks(weakFrameworks.build());
      }

      if (ruleContext.attributes().has("sdk_dylibs", Types.STRING_LIST)) {
        NestedSetBuilder<String> sdkDylibs = NestedSetBuilder.stableOrder();
        sdkDylibs.addAll(ruleContext.attributes().get("sdk_dylibs", Types.STRING_LIST));
        builder.addSdkDylibs(sdkDylibs.build());
      }
    }

    private static void addCompileOptionsFromRuleContext(Builder builder, RuleContext ruleContext)
        throws InterruptedException {
      addCompileOptionsFromRuleContext(builder, ruleContext, /* copts= */ null);
    }

    static void addCompileOptionsFromRuleContext(
        Builder builder, RuleContext ruleContext, Iterable<String> copts)
        throws InterruptedException {
      if (ruleContext.attributes().has("copts", Types.STRING_LIST)) {
        if (copts == null) {
          builder.addCopts(ruleContext.getExpander().withDataLocations().tokenized("copts"));
        } else {
          builder.addCopts(copts);
        }
      }

      if (ruleContext.attributes().has("linkopts", Types.STRING_LIST)) {
        builder.addLinkopts(CppHelper.getLinkopts(ruleContext));
      }

      if (ruleContext.attributes().has("additional_linker_inputs", BuildType.LABEL_LIST)) {
        builder.addLinkInputs(
            ruleContext.getPrerequisiteArtifacts("additional_linker_inputs").list());
      }

      if (ruleContext.attributes().has("defines", Types.STRING_LIST)) {
        builder.addDefines(ruleContext.getExpander().withDataLocations().tokenized("defines"));
      }
    }

    protected static void addModuleOptionsFromRuleContext(
        Builder builder, RuleContext ruleContext) {
      PathFragment packageFragment = ruleContext.getPackageDirectory();
      if (packageFragment != null) {
        builder.setPackageFragment(packageFragment);
      }

      if (ruleContext.attributes().has("enable_modules", Type.BOOLEAN)
          && ruleContext.attributes().get("enable_modules", Type.BOOLEAN)) {
        builder.enableModules();
      }
    }
  }

  private final NestedSet<Artifact> hdrs;
  private final NestedSet<Artifact> textualHdrs;
  private final NestedSet<PathFragment> includes;
  private final NestedSet<PathFragment> sdkIncludes;
  private final NestedSet<String> sdkFrameworks;
  private final NestedSet<String> weakSdkFrameworks;
  private final NestedSet<String> sdkDylibs;
  private final Optional<PathFragment> packageFragment;
  private final ImmutableList<String> copts;
  private final ImmutableList<String> linkopts;
  private final ImmutableList<Artifact> linkInputs;
  private final ImmutableList<String> defines;
  private final boolean enableModules;

  private CompilationAttributes(
      NestedSet<Artifact> hdrs,
      NestedSet<Artifact> textualHdrs,
      NestedSet<PathFragment> includes,
      NestedSet<PathFragment> sdkIncludes,
      NestedSet<String> sdkFrameworks,
      NestedSet<String> weakSdkFrameworks,
      NestedSet<String> sdkDylibs,
      Optional<PathFragment> packageFragment,
      ImmutableList<String> copts,
      ImmutableList<String> linkopts,
      ImmutableList<Artifact> linkInputs,
      ImmutableList<String> defines,
      boolean enableModules) {
    this.hdrs = hdrs;
    this.textualHdrs = textualHdrs;
    this.includes = includes;
    this.sdkIncludes = sdkIncludes;
    this.sdkFrameworks = sdkFrameworks;
    this.weakSdkFrameworks = weakSdkFrameworks;
    this.sdkDylibs = sdkDylibs;
    this.packageFragment = packageFragment;
    this.copts = copts;
    this.linkopts = linkopts;
    this.linkInputs = linkInputs;
    this.defines = defines;
    this.enableModules = enableModules;
  }

  /**
   * Returns the headers to be made available for dependents.
   */
  public NestedSet<Artifact> hdrs() {
    return this.hdrs;
  }

  @StarlarkMethod(name = "hdrs", documented = false, structField = true)
  public Depset hdrsForStarlark() {
    return Depset.of(Artifact.class, hdrs());
  }

  @StarlarkMethod(name = "textual_hdrs", documented = false, structField = true)
  public Depset textualHdrsForStarlark() {
    return Depset.of(Artifact.class, textualHdrs());
  }

  /**
   * Returns the headers that cannot be compiled individually.
   */
  public NestedSet<Artifact> textualHdrs() {
    return this.textualHdrs;
  }

  /**
   * Returns the include paths to be made available for compilation.
   */
  public NestedSet<PathFragment> includes() {
    return this.includes;
  }

  @StarlarkMethod(name = "includes", documented = false, structField = true)
  public Depset includesForStarlark() {
    return Depset.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.COMPILE_ORDER,
            includes().toList().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  @StarlarkMethod(name = "sdk_includes", documented = false, structField = true)
  public Depset sdkIncludesForStarlark() {
    return Depset.of(
        String.class,
        NestedSetBuilder.wrap(
            Order.COMPILE_ORDER,
            sdkIncludes().toList().stream()
                .map(PathFragment::getSafePathString)
                .collect(ImmutableList.toImmutableList())));
  }

  /**
   * Returns the paths for SDK includes.
   */
  public NestedSet<PathFragment> sdkIncludes() {
    return this.sdkIncludes;
  }

  /** Returns the SDK frameworks to link against. */
  public NestedSet<String> sdkFrameworks() {
    return this.sdkFrameworks;
  }

  @StarlarkMethod(name = "sdk_frameworks", documented = false, structField = true)
  public Depset sdkFrameworksForStarlark() {
    return Depset.of(String.class, sdkFrameworks);
  }

  @StarlarkMethod(name = "weak_sdk_frameworks", documented = false, structField = true)
  public Depset weakSdkFrameworksForStarlark() {
    return Depset.of(String.class, weakSdkFrameworks);
  }

  /** Returns the SDK frameworks to be linked weakly. */
  public NestedSet<String> weakSdkFrameworks() {
    return this.weakSdkFrameworks;
  }

  @StarlarkMethod(name = "sdk_dylibs", documented = false, structField = true)
  public Depset sdkDylibsForStarlark() {
    return Depset.of(String.class, sdkDylibs);
  }

  /**
   * Returns the SDK Dylibs to link against.
   */
  public NestedSet<String> sdkDylibs() {
    return this.sdkDylibs;
  }

  @StarlarkMethod(
      name = "header_search_paths",
      documented = false,
      parameters = {
        @Param(name = "genfiles_dir", positional = false, named = true),
      })
  public Depset headerSearchPathsForStarlark(String genfilesDir) {
    return Depset.of(
        String.class,
        NestedSetBuilder.<String>stableOrder()
            .addAll(
                headerSearchPaths(PathFragment.create(genfilesDir)).toList().stream()
                    .map(PathFragment::toString)
                    .collect(ImmutableList.toImmutableList()))
            .build());
  }

  /**
   * Returns the exec paths of all header search paths that should be added to this target and
   * dependers on this target, obtained from the {@code includes} attribute.
   */
  public NestedSet<PathFragment> headerSearchPaths(PathFragment genfilesFragment) {
    NestedSetBuilder<PathFragment> paths = NestedSetBuilder.stableOrder();
    if (packageFragment.isPresent()) {
      PathFragment packageFrag = packageFragment.get();
      PathFragment genfilesFrag = genfilesFragment.getRelative(packageFrag);
      for (PathFragment include : includes().toList()) {
        if (!include.isAbsolute()) {
          paths.add(packageFrag.getRelative(include));
          paths.add(genfilesFrag.getRelative(include));
        }
      }
    }
    return paths.build();
  }

  /**
   * Returns the compile-time options.
   */
  public ImmutableList<String> copts() {
    return this.copts;
  }

  @StarlarkMethod(name = "copts", documented = false, structField = true)
  public Sequence<String> getCoptsForStarlark() {
    return StarlarkList.immutableCopyOf(copts());
  }

  /**
   * Returns the link-time options.
   */
  public ImmutableList<String> linkopts() {
    return this.linkopts;
  }

  /** Returns the additional link inputs. */
  public ImmutableList<Artifact> linkInputs() {
    return this.linkInputs;
  }

  @StarlarkMethod(name = "defines", documented = false, structField = true)
  public Sequence<String> getDefinesForStarlark() {
    return StarlarkList.immutableCopyOf(defines());
  }

  /** Returns the defines. */
  public ImmutableList<String> defines() {
    return this.defines;
  }

  /**
   * Returns whether this target uses language features that require clang modules, such as
   * {@literal @}import.
   */
  @StarlarkMethod(name = "enable_modules", documented = false, structField = true)
  public boolean enableModules() {
    return this.enableModules;
  }
}
