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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.TOP_LEVEL_MODULE_MAP;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Provides a way to access attributes that are common to all compilation rules.
 */
// TODO(bazel-team): Delete and move into support-specific attributes classes once ObjcCommon is
// gone.
final class CompilationAttributes {
  static class Builder {
    private final NestedSetBuilder<Artifact> hdrs = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> textualHdrs = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<PathFragment> includes = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<PathFragment> sdkIncludes = NestedSetBuilder.stableOrder();
    private final ImmutableList.Builder<String> copts = ImmutableList.builder();
    private final ImmutableList.Builder<String> linkopts = ImmutableList.builder();
    private final ImmutableList.Builder<String> defines = ImmutableList.builder();
    private final NestedSetBuilder<CppModuleMap> moduleMapsForDirectDeps =
        NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<SdkFramework> sdkFrameworks = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<SdkFramework> weakSdkFrameworks = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<String> sdkDylibs = NestedSetBuilder.stableOrder();
    private Optional<PathFragment> packageFragment = Optional.absent();
    private boolean enableModules;

    /**
     * Adds the default values available through the rule's context.
     */
    static Builder fromRuleContext(RuleContext ruleContext) {
      Builder builder = new Builder();

      addHeadersFromRuleContext(builder, ruleContext);
      addIncludesFromRuleContext(builder, ruleContext);
      addSdkAttributesFromRuleContext(builder, ruleContext);
      addCompileOptionsFromRuleContext(builder, ruleContext);
      addModuleOptionsFromRuleContext(builder, ruleContext);

      return builder;
    }

    /**
     * Adds headers to be made available for dependents.
     */
    public Builder addHdrs(NestedSet<Artifact> hdrs) {
      this.hdrs.addTransitive(hdrs);
      return this;
    }

    /**
     * Adds headers that cannot be compiled individually.
     */
    public Builder addTextualHdrs(NestedSet<Artifact> textualHdrs) {
      this.textualHdrs.addTransitive(textualHdrs);
      return this;
    }

    /**
     * Adds include paths to be made available for compilation.
     */
    public Builder addIncludes(NestedSet<PathFragment> includes) {
      this.includes.addTransitive(includes);
      return this;
    }

    /**
     * Adds paths for SDK includes.
     */
    public Builder addSdkIncludes(NestedSet<PathFragment> sdkIncludes) {
      this.sdkIncludes.addTransitive(sdkIncludes);
      return this;
    }

    /**
     * Adds compile-time options.
     */
    public Builder addCopts(Iterable<String> copts) {
      this.copts.addAll(copts);
      return this;
    }

    /**
     * Adds link-time options.
     */
    public Builder addLinkopts(Iterable<String> linkopts) {
      this.linkopts.addAll(linkopts);
      return this;
    }

    /** Adds defines. */
    public Builder addDefines(Iterable<String> defines) {
      this.defines.addAll(defines);
      return this;
    }

    /**
     * Adds clang module maps for direct dependencies of the rule. These are needed to generate
     * module maps.
     */
    public Builder addModuleMapsForDirectDeps(NestedSet<CppModuleMap> moduleMapsForDirectDeps) {
      this.moduleMapsForDirectDeps.addTransitive(moduleMapsForDirectDeps);
      return this;
    }

    /**
     * Adds SDK frameworks to link against.
     */
    public Builder addSdkFrameworks(NestedSet<SdkFramework> sdkFrameworks) {
      this.sdkFrameworks.addTransitive(sdkFrameworks);
      return this;
    }

    /**
     * Adds SDK frameworks to be linked weakly.
     */
    public Builder addWeakSdkFrameworks(NestedSet<SdkFramework> weakSdkFrameworks) {
      this.weakSdkFrameworks.addTransitive(weakSdkFrameworks);
      return this;
    }

    /**
     * Adds SDK Dylibs to link against.
     */
    public Builder addSdkDylibs(NestedSet<String> sdkDylibs) {
      this.sdkDylibs.addTransitive(sdkDylibs);
      return this;
    }

    /**
     * Sets the package path from which to base the header search paths.
     */
    public Builder setPackageFragment(PathFragment packageFragment) {
      Preconditions.checkState(
          !this.packageFragment.isPresent(),
          "packageFragment is already set to %s",
          this.packageFragment);
      this.packageFragment = Optional.of(packageFragment);
      return this;
    }

    /**
     * Enables the usage of clang modules maps during compilation.
     */
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
          this.defines.build(),
          this.moduleMapsForDirectDeps.build(),
          this.enableModules);
    }

    private static void addHeadersFromRuleContext(Builder builder, RuleContext ruleContext) {
      if (ruleContext.attributes().has("hdrs", BuildType.LABEL_LIST)) {
        NestedSetBuilder<Artifact> headers = NestedSetBuilder.stableOrder();
        for (Pair<Artifact, Label> header : CcCommon.getHeaders(ruleContext)) {
          headers.add(header.first);
        }
        builder.addHdrs(headers.build());
      }

      if (ruleContext.attributes().has("textual_hdrs", BuildType.LABEL_LIST)) {
        builder.addTextualHdrs(
            PrerequisiteArtifacts.nestedSet(ruleContext, "textual_hdrs", Mode.TARGET));
      }
    }

    private static void addIncludesFromRuleContext(Builder builder, RuleContext ruleContext) {
      if (ruleContext.attributes().has("includes", Type.STRING_LIST)) {
        NestedSetBuilder<PathFragment> includes = NestedSetBuilder.stableOrder();
        includes.addAll(
            Iterables.transform(
                ruleContext.attributes().get("includes", Type.STRING_LIST), PathFragment::create));
        builder.addIncludes(includes.build());
      }

      if (ruleContext.attributes().has("sdk_includes", Type.STRING_LIST)) {
        NestedSetBuilder<PathFragment> sdkIncludes = NestedSetBuilder.stableOrder();
        sdkIncludes.addAll(
            Iterables.transform(
                ruleContext.attributes().get("sdk_includes", Type.STRING_LIST),
                PathFragment::create));
        builder.addSdkIncludes(sdkIncludes.build());
      }
    }

    private static void addSdkAttributesFromRuleContext(Builder builder, RuleContext ruleContext) {
      if (ruleContext.attributes().has("sdk_frameworks", Type.STRING_LIST)) {
        NestedSetBuilder<SdkFramework> frameworks = NestedSetBuilder.stableOrder();
        for (String explicit : ruleContext.attributes().get("sdk_frameworks", Type.STRING_LIST)) {
          frameworks.add(new SdkFramework(explicit));
        }
        builder.addSdkFrameworks(frameworks.build());
      }

      if (ruleContext.attributes().has("weak_sdk_frameworks", Type.STRING_LIST)) {
        NestedSetBuilder<SdkFramework> weakFrameworks = NestedSetBuilder.stableOrder();
        for (String frameworkName :
            ruleContext.attributes().get("weak_sdk_frameworks", Type.STRING_LIST)) {
          weakFrameworks.add(new SdkFramework(frameworkName));
        }
        builder.addWeakSdkFrameworks(weakFrameworks.build());
      }

      if (ruleContext.attributes().has("sdk_dylibs", Type.STRING_LIST)) {
        NestedSetBuilder<String> sdkDylibs = NestedSetBuilder.stableOrder();
        sdkDylibs.addAll(ruleContext.attributes().get("sdk_dylibs", Type.STRING_LIST));
        builder.addSdkDylibs(sdkDylibs.build());
      }
    }

    private static void addCompileOptionsFromRuleContext(Builder builder, RuleContext ruleContext) {
      if (ruleContext.attributes().has("copts", Type.STRING_LIST)) {
        builder.addCopts(ruleContext.getExpander().withDataLocations().tokenized("copts"));
      }

      if (ruleContext.attributes().has("linkopts", Type.STRING_LIST)) {
        builder.addLinkopts(ruleContext.getExpander().withDataLocations().tokenized("linkopts"));
      }

      if (ruleContext.attributes().has("defines", Type.STRING_LIST)) {
        builder.addDefines(ruleContext.getExpander().withDataLocations().tokenized("defines"));
      }
    }

    private static void addModuleOptionsFromRuleContext(Builder builder, RuleContext ruleContext) {
      NestedSetBuilder<CppModuleMap> moduleMaps = NestedSetBuilder.stableOrder();
      ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(ruleContext);
      if (objcConfiguration.moduleMapsEnabled()) {
        // Make sure all dependencies that have headers are included here. If a module map is
        // missing, its private headers will be treated as public!
        if (ruleContext.attributes().has("deps", BuildType.LABEL_LIST)) {
          Iterable<ObjcProvider> providers =
              ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR);
          for (ObjcProvider provider : providers) {
            moduleMaps.addTransitive(provider.get(TOP_LEVEL_MODULE_MAP));
          }
        }
      }

      builder.addModuleMapsForDirectDeps(moduleMaps.build());

      PathFragment packageFragment =
          ruleContext.getLabel().getPackageIdentifier().getSourceRoot();
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
  private final NestedSet<SdkFramework> sdkFrameworks;
  private final NestedSet<SdkFramework> weakSdkFrameworks;
  private final NestedSet<String> sdkDylibs;
  private final Optional<PathFragment> packageFragment;
  private final ImmutableList<String> copts;
  private final ImmutableList<String> linkopts;
  private final ImmutableList<String> defines;
  private final NestedSet<CppModuleMap> moduleMapsForDirectDeps;
  private final boolean enableModules;

  private CompilationAttributes(
      NestedSet<Artifact> hdrs,
      NestedSet<Artifact> textualHdrs,
      NestedSet<PathFragment> includes,
      NestedSet<PathFragment> sdkIncludes,
      NestedSet<SdkFramework> sdkFrameworks,
      NestedSet<SdkFramework> weakSdkFrameworks,
      NestedSet<String> sdkDylibs,
      Optional<PathFragment> packageFragment,
      ImmutableList<String> copts,
      ImmutableList<String> linkopts,
      ImmutableList<String> defines,
      NestedSet<CppModuleMap> moduleMapsForDirectDeps,
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
    this.defines = defines;
    this.moduleMapsForDirectDeps = moduleMapsForDirectDeps;
    this.enableModules = enableModules;
  }

  /**
   * Returns the headers to be made available for dependents.
   */
  public NestedSet<Artifact> hdrs() {
    return this.hdrs;
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

  /**
   * Returns the paths for SDK includes.
   */
  public NestedSet<PathFragment> sdkIncludes() {
    return this.sdkIncludes;
  }

  /**
   * Returns the SDK frameworks to link against.
   */
  public NestedSet<SdkFramework> sdkFrameworks() {
    return this.sdkFrameworks;
  }

  /**
   * Returns the SDK frameworks to be linked weakly.
   */
  public NestedSet<SdkFramework> weakSdkFrameworks() {
    return this.weakSdkFrameworks;
  }

  /**
   * Returns the SDK Dylibs to link against.
   */
  public NestedSet<String> sdkDylibs() {
    return this.sdkDylibs;
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
      for (PathFragment include : includes()) {
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

  /**
   * Returns the link-time options.
   */
  public ImmutableList<String> linkopts() {
    return this.linkopts;
  }

  /** Returns the defines. */
  public ImmutableList<String> defines() {
    return this.defines;
  }

  /**
   * Returns the clang module maps of direct dependencies of this rule. These are needed to generate
   * this rule's module map.
   */
  public NestedSet<CppModuleMap> moduleMapsForDirectDeps() {
    return this.moduleMapsForDirectDeps;
  }

  /**
   * Returns whether this target uses language features that require clang modules, such as
   * {@literal @}import.
   */
  public boolean enableModules() {
    return this.enableModules;
  }
}
