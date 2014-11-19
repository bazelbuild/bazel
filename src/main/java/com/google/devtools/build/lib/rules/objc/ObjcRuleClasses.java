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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.BaseRuleClasses;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

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

  static IntermediateArtifacts intermediateArtifacts(RuleContext ruleContext) {
    return new IntermediateArtifacts(
        ruleContext.getAnalysisEnvironment(), ruleContext.getBinOrGenfilesDirectory(),
        ruleContext.getLabel());
  }

  static Artifact artifactByAppendingToBaseName(RuleContext context, String suffix) {
    return artifactByAppendingToRootRelativePath(
        context, context.getLabel().toPathFragment(), suffix);
  }

  static ObjcActionsBuilder actionsBuilder(RuleContext ruleContext) {
    return new ObjcActionsBuilder(
        ruleContext,
        intermediateArtifacts(ruleContext),
        ObjcRuleClasses.objcConfiguration(ruleContext),
        ruleContext.getConfiguration(),
        ruleContext);
  }

  static ObjcConfiguration objcConfiguration(RuleContext ruleContext) {
    return ruleContext.getConfiguration().getFragment(ObjcConfiguration.class);
  }

  @VisibleForTesting
  static final Iterable<SdkFramework> AUTOMATIC_SDK_FRAMEWORKS = ImmutableList.of(
      new SdkFramework("Foundation"), new SdkFramework("UIKit"));

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
          ${SYNOPSIS}
          Subject to <a href="#make_variables">"Make variable"</a> substitution and
          <a href="#sh-tokenization">Bourne shell tokenization</a>.
          These flags will only apply to this target, and not those upon which
          it depends, or those which depend on it.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("copts", STRING_LIST))
          .build();
    }
  }

  /**
   * Iff a file matches this type, it is considered to use C++.
   */
  static final FileType CPP_SOURCES = FileType.of(".cc", ".cpp", ".mm", ".cxx");

  private static final FileType NON_CPP_SOURCES = FileType.of(".m", ".c");

  static final FileTypeSet SRCS_TYPE = FileTypeSet.of(NON_CPP_SOURCES, CPP_SOURCES);

  static final FileTypeSet NON_ARC_SRCS_TYPE = FileTypeSet.of(FileType.of(".m", ".mm"));

  // TODO(bazel-team): Remove .pch when depot cleanup is done
  static final FileTypeSet HDRS_TYPE =
      FileTypeSet.of(CPP_SOURCES, NON_CPP_SOURCES, FileType.of(".h", ".hh", ".pch"));

  static final FileTypeSet PLIST_TYPE = FileTypeSet.of(FileType.of(".plist"));

  static final FileTypeSet STORYBOARD_TYPE = FileTypeSet.of(FileType.of(".storyboard"));

  static final FileType XIB_TYPE = FileType.of(".xib");

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
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(hdrs) -->
          The list of Objective-C files that are included as headers by source
          files in this rule or by users of this library.
          ${SYNOPSIS}
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("hdrs", LABEL_LIST)
              .direct_compile_time_input()
              .allowedFileTypes(HDRS_TYPE))
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(includes) -->
          List of <code>#include/#import</code> search paths to add to this target
          and all depending targets. This is to support third party and
          open-sourced libraries that do not specify the entire workspace path in
          their <code>#import/#include</code> statements.
          <p>
          The paths are interpreted relative to the package directory, and the
          genfiles and bin roots (e.g. <code>blaze-genfiles/pkg/includedir</code>
          and <code>blaze-out/pkg/includedir</code>) are included in addition to the
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
          .add(attr("asset_catalogs", LABEL_LIST).legacyAllowAnyFileType()
              .direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(strings) -->
          Files which are plists of strings, often localizable. These files
          are converted to binary plists (if they are not already) and placed
          in the bundle root of the final package. If this file's immediate
          containing directory is named *.lproj (e.g. en.lproj, Base.lproj), it
          will be placed under a directory of that name in the final bundle.
          This allows for localizable strings.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("strings", LABEL_LIST).legacyAllowAnyFileType()
              .direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(xibs) -->
          Files which are .xib resources, possibly localizable. These files are
          compiled to .nib files and placed the bundle root of the final
          package. If this file's immediate containing directory is named
          *.lproj (e.g. en.lproj, Base.lproj), it will be placed under a
          directory of that name in the final bundle. This allows for
          localizable UI.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("xibs", LABEL_LIST)
              .direct_compile_time_input()
              .allowedFileTypes(XIB_TYPE))
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(storyboards) -->
          Files which are .storyboard resources, possibly localizable. These
          files are compiled to .storyboardc directories, which are placed in
          the bundle root of the final package. If the storyboards's immediate
          containing directory is named *.lproj (e.g. en.lproj, Base.lproj), it
          will be placed under a directory of that name in the final bundle.
          This allows for localizable UI.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("storyboards", LABEL_LIST)
              .allowedFileTypes(STORYBOARD_TYPE))
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
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(weak_sdk_frameworks) -->
          Names of SDK frameworks to weakly link with. For instance,
          "MediaAccessibility". In difference to regularly linked SDK
          frameworks, symbols from weakly linked frameworks do not cause an
          error if they are not present at runtime.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("weak_sdk_frameworks", STRING_LIST))
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(sdk_dylibs) -->
          Names of SDK .dylib libraries to link with. For instance, "libz" or
          "libarchive". "libc++" is included automatically if the binary has
          any C++ or Objective-C++ sources in its dependency tree. When linking
          a binary, all libraries named in that binary's transitive dependency
          graph are used.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("sdk_dylibs", STRING_LIST))
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(resources) -->
          Files to include in the final application bundle. They are not
          processed or compiled in any way besides the processing done by the
          rules that actually generate them. These files are placed in the root
          of the bundle (e.g. Payload/foo.app/...) in most cases. However, if
          they appear to be localized (i.e. are contained in a directory called
          *.lproj), they will be placed in a directory of the same name in the
          app bundle.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("resources", LABEL_LIST).legacyAllowAnyFileType().direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_rule).ATTRIBUTE(datamodels) -->
          Files that comprise the data models of the final linked binary.
          Each file must have a containing directory named *.xcdatamodel, which
          is usually contained by another *.xcdatamodeld (note the added d)
          directory.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("datamodels", LABEL_LIST).legacyAllowAnyFileType()
              .direct_compile_time_input())
          .add(attr("$xcodegen", LABEL).cfg(HOST).exec()
              .value(env.getLabel("//tools/objc:xcodegen")))
          .add(attr("$plmerge", LABEL).cfg(HOST).exec()
              .value(env.getLabel("//tools/objc:plmerge")))
          .add(attr("$momczip_deploy", LABEL).cfg(HOST)
              .value(env.getLabel("//tools/objc:momczip_deploy.jar")))
          .add(attr("$actooloribtoolzip_deploy", LABEL).cfg(HOST)
              .value(env.getLabel("//tools/objc:actooloribtoolzip_deploy.jar")))
          .build();
    }
  }

  /**
   * Object that supplies tools used by all rules which have the helper tools common to most rule
   * implementations.
   */
  static final class Tools {
    private final RuleContext ruleContext;
  
    Tools(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
    }
  
    Artifact actooloribtoolzipDeployJar() {
      return ruleContext.getPrerequisiteArtifact("$actooloribtoolzip_deploy", Mode.HOST);
    }
  
    Artifact momczipDeployJar() {
      return ruleContext.getPrerequisiteArtifact("$momczip_deploy", Mode.HOST);
    }
  
    FilesToRunProvider xcodegen() {
      return ruleContext.getExecutablePrerequisite("$xcodegen", Mode.HOST);
    }
  
    FilesToRunProvider plmerge() {
      return ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST);
    }
  }
}
