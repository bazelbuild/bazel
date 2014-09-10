// Copyright 2014 Google Inc. All rights reserved.
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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.NON_ARC_SRCS;
import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.SRCS;
import static com.google.devtools.build.xcode.common.BuildOptionsUtil.DEFAULT_OPTIONS_NAME;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.AnalysisUtils;
import com.google.devtools.build.lib.view.BaseRuleClasses;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

import java.util.List;

/**
 * Shared utility code for Objective-C rules.
 */
public class ObjcRuleClasses {
  public static final SafeImplicitOutputsFunction PBXPROJ =
      fromTemplates("%{name}.xcodeproj/project.pbxproj");

  private ObjcRuleClasses() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Returns the artifact corresponding to the pbxproj file for an objc_binary or objc_library
   * target.
   */
  static Artifact pbxprojArtifact(RuleContext context) {
    PathFragment labelPath = context.getLabel().toPathFragment();
    return context.getAnalysisEnvironment().getDerivedArtifact(
        labelPath.replaceName(labelPath.getBaseName() + ".xcodeproj")
            .getRelative("project.pbxproj"),
        context.getBinOrGenfilesDirectory());
  }

  /**
   * Returns a derived Artifact by appending a String to a root-relative path. This is similar to
   * {@link RuleContext#getRelatedArtifact(PathFragment, String)}, except the existing extension is
   * not removed.
   */
  static Artifact artifactByAppendingToRootRelativePath(
      RuleContext ruleContext, PathFragment path, String suffix) {
    return ruleContext.getAnalysisEnvironment().getDerivedArtifact(
        path.replaceName(path.getBaseName() + suffix),
        ruleContext.getBinOrGenfilesDirectory());
  }

  static Artifact artifactByAppendingToBaseName(RuleContext context, String suffix) {
    return artifactByAppendingToRootRelativePath(
        context, context.getLabel().toPathFragment(), suffix);
  }

  /**
   * Returns the artifact corresponding to the pbxproj control file, which specifies the information
   * required to generate the Xcode project file.
   */
  static Artifact pbxprojControlArtifact(RuleContext context) {
    return artifactByAppendingToBaseName(context, ".xcodeproj-control");
  }

  /**
   * Returns the artifact corresponding to the bundlemerge control file for creating the application
   * bundle.
   */
  static Artifact bundleMergeControlArtifact(RuleContext context) {
    return artifactByAppendingToBaseName(context, ".ipa-control");
  }

  /**
   * Returns the artifact which is the output of running actool on all the asset catalogs used by
   * a binary.
   */
  static Artifact actoolOutputZip(RuleContext context) {
    return artifactByAppendingToBaseName(context, ".actool.zip");
  }

  /**
   * Returns the artifact which is the output of building an entire xcdatamodel[d] made of artifacts
   * specified by a single rule.
   * @param context the rule that specifies the {@code data_models} attribute
   * @param containerDir the containing *.xcdatamodeld or *.xcdatamodel directory
   * @return the artifact for the zipped up compilation results.
   */
  static Artifact compiledMomZipArtifact(RuleContext context, PathFragment containerDir) {
    return artifactByAppendingToBaseName(context,
        "/" + FileSystemUtils.replaceExtension(containerDir, ".zip").getBaseName());
  }

  @VisibleForTesting
  static final Iterable<SdkFramework> AUTOMATIC_SDK_FRAMEWORKS = ImmutableList.of(
      new SdkFramework("Foundation"), new SdkFramework("UIKit"));

  /**
   * Returns the value of the sdk_frameworks attribute plus frameworks that are included
   * automatically. 
   */
  static Iterable<SdkFramework> sdkFrameworks(RuleContext context) {
    ImmutableSet.Builder<SdkFramework> result = new ImmutableSet.Builder<>();
    result.addAll(AUTOMATIC_SDK_FRAMEWORKS);
    for (String explicit : context.attributes().get("sdk_frameworks", Type.STRING_LIST)) {
      result.add(new SdkFramework(explicit));
    }
    return result.build();
  }

  static Iterable<String> copts(RuleContext context) {
    if (context.attributes().getAttributeDefinition("copts") == null) {
      return ImmutableList.of();
    } else {
      return context.getTokenizedStringListAttr("copts");
    }
  }

  /**
   * The artifact for the .o file that should be generated when compiling the {@code source}
   * artifact.
   */
  static Artifact objFile(RuleContext context, Artifact source) {
    return context.getRelatedArtifact(
        AnalysisUtils.getUniqueDirectory(context.getLabel(), new PathFragment("_objcs"))
            .getRelative(source.getRootRelativePath()), ".o");
  }

  static Iterable<Artifact> objFiles(final RuleContext context, Iterable<Artifact> sources) {
    return Iterables.transform(sources,
        new Function<Artifact, Artifact>() {
          @Override
          public Artifact apply(Artifact source) {
            return objFile(context, source);
          }
        });
  }

  static Optional<PathFragment> outputAFilePackageRelativePath(AttributeMap map) {
    if (Iterables.isEmpty(SRCS.get(map)) && Iterables.isEmpty(NON_ARC_SRCS.get(map))) {
      return Optional.absent();
    }
    PathFragment labelName = new PathFragment(map.getLabel().getName());
    return Optional.of(
        labelName
            .getParentDirectory()
            .getRelative(String.format("lib%s.a", labelName.getBaseName())));
  }

  /**
   * The artifact generated by objc_binary and objc_library targets that have at least one source
   * file.
   */
  static Optional<Artifact> outputAFile(final RuleContext context) {
    return outputAFilePackageRelativePath(context.attributes()).transform(
        new Function<PathFragment, Artifact>() {
          @Override
          public Artifact apply(PathFragment packageRelativePath) {
            return context.getAnalysisEnvironment().getDerivedArtifact(
                context.getLabel().getPackageFragment().getRelative(packageRelativePath),
                context.getBinOrGenfilesDirectory());
          }
        });
  }

  static Optional<Artifact> pchFile(RuleContext ruleContext) {
    if (ruleContext.attributes().getAttributeDefinition("pch") == null) {
      return Optional.absent();
    } else {
      return Optional.fromNullable(ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET));
    }
  }

  /**
   * Gets the providers for the {@code deps} of the given rule, or an empty sequence if the rule
   * or rule type does not have a {@code deps} attribute.
   */
  static <P extends TransitiveInfoProvider>
      Iterable<P> deps(RuleContext context, Class<P> providerClass) {
    if (context.attributes().getAttributeDefinition("deps") == null) {
      return ImmutableList.of();
    } else {
      return context.getPrerequisites("deps", Mode.TARGET, providerClass);
    }
  }

  /**
   * Returns the value of the {@code includes} attribute, where each returned path is under the
   * path corresponding to the current package, and each entry in the attribute results in three
   * items in the returned sequence: one is rooted in the actual client, one is rooted in genfiles,
   * and one is rooted in bin.
   */
  static Iterable<PathFragment> includes(RuleContext context) {
    ImmutableList.Builder<PathFragment> includes = new ImmutableList.Builder<>();
    if (context.attributes().getAttributeDefinition("includes") != null) {
      PathFragment packageFragment = context.getLabel().getPackageFragment();
      List<PathFragment> rootFragments = ImmutableList.of(
          packageFragment,
          context.getConfiguration().getGenfilesFragment().getRelative(packageFragment),
          context.getConfiguration().getBinFragment().getRelative(packageFragment));

      for (String include : context.attributes().get("includes", STRING_LIST)) {
        // TODO(bazel-team): ignore items that are absolute paths, and report an error for them.
        for (PathFragment rootFragment : rootFragments) {
          includes.add(rootFragment.getRelative(include).normalize());
        }
      }
    }
    return includes.build();
  }

  /**
   * Returns build options information on the given rule, automatically giving the default value if
   * it is not specified or allowed on this rule type.
   */
  static OptionsProvider options(RuleContext context) {
    OptionsProvider options = null;
    if (context.attributes().getAttributeDefinition("options") != null) {
      options = context.getPrerequisite("options", Mode.TARGET, OptionsProvider.class);
    }
    if (options == null) {
      options = new OptionsProvider(DEFAULT_OPTIONS_NAME, ImmutableList.<String>of());
    }
    return options;
  }

  /**
   * Attributes for {@code objc_*} rules that have compiler (and in the future, possibly linker)
   * options
   */
  @BlazeRule(name = "$objc_opts_rule",
      type = RuleClassType.ABSTRACT)
  public static class ObjcOptsRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          /* <!-- #BLAZE_RULE($objc_opts_rule).ATTRIBUTE(copts) -->
          Extra flags to pass to the compiler.
          <i>(List of strings; optional; subject to
            <a href="#make_variables">"Make variable"</a> substitution and
            <a href="#sh-tokenization">Bourne shell tokenization</a>)</i>
          These flags will only apply to this target, and not those upon which
          it depends, or those which depend on it.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("copts", STRING_LIST))
          .build();
    }
  }

  /**
   * Attributes for {@code objc_*} rules that have compilable sources.
   */
  @BlazeRule(name = "$objc_sources_rule",
      type = RuleClassType.ABSTRACT)
  public static class ObjcSourcesRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          /* <!-- #BLAZE_RULE($objc_sources_rule).ATTRIBUTE(srcs) -->
          The list of Objective-C files that are processed to create the
          library target.
          <i>(List of <a href="build-ref.html#labels">labels</a>; required)</i>
          These are your checked-in source files, plus any generated files.
          These are compiled into .o files with Clang, so headers should not go
          here (see the hdrs attribute).
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("srcs", LABEL_LIST)
              .direct_compile_time_input()
              // TODO(bazel-team): Get .mm and .cc file types compiling.
              .allowedFileTypes(FileType.of("m"), FileType.of("c")))
          /* <!-- #BLAZE_RULE($objc_sources_rule).ATTRIBUTE(non_arc_srcs) -->
          The list of Objective-C files that are processed to create the
          library target that DO NOT use ARC.
          <i>(List of <a href="build-ref.html#labels">labels</a>; required)</i>
          The files in this attribute are treated very similar to those in the
          srcs attribute, but are compiled without ARC enabled.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("non_arc_srcs", LABEL_LIST)
              .direct_compile_time_input()
              // TODO(bazel-team): Get .mm files compiling.
              .allowedFileTypes(FileType.of("m")))
          /* <!-- #BLAZE_RULE($objc_sources_rule).ATTRIBUTE(pch) -->
          Header file to prepend to every source file being compiled (both arc
          and non-arc). Note that the file will not be precompiled - this is
          simply a convenience, not a build-speed enhancement.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("pch", LABEL)
              .direct_compile_time_input()
              .allowedFileTypes(FileType.of("pch")))
          /* <!-- #BLAZE_RULE($objc_sources_rule).ATTRIBUTE(options) -->
          An {@code objc_options} target which defines an Xcode build
          configuration profile.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("options", LABEL)
              .allowedFileTypes()
              .allowedRuleClasses("objc_options"))
          .build();
    }
  }

  /**
   * Common attributes for {@code objc_*} rules.
   */
  @BlazeRule(name = "$objc_base_rule",
      type = RuleClassType.ABSTRACT,
      ancestors = { BaseRuleClasses.RuleBase.class })
  public static class ObjcBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(deps) -->
          The list of {@code objc_library} and {@code objc_import}
          targets that are linked together to form the final bundle.
          <i>(List of <a href="build-ref.html#labels">labels</a>; required)</i>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .override(attr("deps", LABEL_LIST)
              .direct_compile_time_input()
              .allowedRuleClasses("objc_library", "objc_import")
              .allowedFileTypes())
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(hdrs) -->
          The list of Objective-C files that are included as headers by source
          files in this rule or by users of this library.
          <i>(List of <a href="build-ref.html#labels">labels</a>; required)</i>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("hdrs", LABEL_LIST)
              .direct_compile_time_input()
              .allowedFileTypes(FileType.of("m"), FileType.of("h")))
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(includes) -->
          List of {@code #include/#import} search paths to add to this target
          and all depending targets. This is to support third party and
          open-sourced libraries that do not specify the entire google3 path in
          their {@code #import/#include} statements.
          <p>
          The paths are interpreted relative to the package directory, and the
          genfiles and bin roots (e.g. {@code blaze-genfiles/pkg/includedir}
          and {@code blaze-out/pkg/includedir}) are included in addition to the
          actual client root.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("includes", Type.STRING_LIST))
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(asset_catalogs) -->
          Files that comprise the asset catalogs of the final linked binary.
          Each file must have a containing directory named *.xcassets. This
          containing directory becomes the root of one of the asset catalogs
          linked with any binary that depends directly or indirectly on this
          target.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("asset_catalogs", LABEL_LIST)
              .direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(strings) -->
          Files which are plists of strings, often localizable. These files
          are converted to binary plists (if they are not already) and placed
          in the bundle root of the final package. If this file's immediate
          containing directory is named *.lproj, it will be placed under a
          directory of that name in the final bundle. This allows for
          localizable strings.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("strings", LABEL_LIST)
              .direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(xibs) -->
          Files which are .xib resources, possibly localizable. These files are
          compiled to .nib files and placed the bundle root of the final
          package. If this file's immediate containing directory is named
          *.lproj, it will be placed under a directory of that name in the
          final bundle. This allows for localizable UI.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("xibs", LABEL_LIST)
              .direct_compile_time_input()
              .allowedFileTypes(FileType.of("xib")))
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(sdk_frameworks) -->
          Names of SDK frameworks to link with. For instance, "XCTest" or
          "Cocoa". "UIKit" and "Foundation" are always included and do not mean
          anything if you include them.
          When linking a library, only those frameworks named in that library's
          sdk_frameworks attribute are linked in. When linking a binary, all
          SDK frameworks named in that binary's transitive dependency graph are
          used.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("sdk_frameworks", STRING_LIST))
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(resources) -->
          Files to include in the final application bundle. They are not
          processed or compiled in any way besides the processing done by the
          rules that actually generate them. These files are placed in the root
          of the bundle (e.g. Payload/foo.app/...) in most cases. However, if
          they appear to be localized (i.e. are contained in a directory called
          *.lproj), they will be placed in a directory of the same name in the
          app bundle.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("resources", LABEL_LIST).direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(datamodels) -->
          Files that comprise the data models of the final linked binary.
          Each file must have a containing directory named *.xcdatamodel, which
          is usually contained by another *.xcdatamodeld (note the added d)
          directory.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("datamodels", LABEL_LIST)
              .direct_compile_time_input())
          .add(attr("$xcodegen", LABEL).cfg(HOST).exec()
              .value(env.getLabel("//tools/objc:xcodegen")))
          .add(attr("$plmerge", LABEL).cfg(HOST).exec()
              .value(env.getLabel("//tools/objc:plmerge")))
          .add(attr("$momczip_deploy", LABEL).cfg(HOST)
              .value(env.getLabel("//tools/objc:momczip_deploy.jar")))
          .build();
    }
  }
}
