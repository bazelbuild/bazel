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
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BaseRuleClasses.BaseRule;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Shared utility code for Objective-C rules.
 */
public class ObjcRuleClasses {

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
        ruleContext.getLabel(), /*archiveFileNameSuffix=*/"");
  }

  /**
   * Returns a {@link IntermediateArtifacts} to be used to compile and link the ObjC source files
   * in {@code j2ObjcSource}.
   */
  static IntermediateArtifacts j2objcIntermediateArtifacts(RuleContext ruleContext,
      J2ObjcSource j2ObjcSource) {
    // We need to append "_j2objc" to the name of the generated archive file to distinguish it from
    // the C/C++ archive file created by proto_library targets with attribute cc_api_version
    // specified.
    return new IntermediateArtifacts(
        ruleContext.getAnalysisEnvironment(),
        ruleContext.getConfiguration().getBinDirectory(),
        j2ObjcSource.getTargetLabel(),
        /*archiveFileNameSuffix=*/"_j2objc");
  }

  /**
   * Returns a {@link J2ObjcSrcsProvider} with J2ObjC-generated ObjC file information from the
   * current rule, and from rules that can be reached transitively through the "deps" attribute.
   *
   * @param ruleContext the rule context of the current rule
   * @param currentSource J2ObjC-generated ObjC file information from the current rule to contribute
   *     to the returned provider
   * @return a {@link J2ObjcSrcsProvider} containing {@code currentSources} and source information
   *         from the transitive closure.
   */
  public static J2ObjcSrcsProvider j2ObjcSrcsProvider(RuleContext ruleContext,
      J2ObjcSource currentSource) {
    return j2ObjcSrcsProvider(ruleContext, Optional.of(currentSource));
  }

  /**
   * Returns a {@link J2ObjcSrcsProvider} with J2ObjC-generated ObjC file information from rules
   * that can be reached transitively through the "deps" attribute.
   *
   * @param ruleContext the rule context of the current rule
   * @return a {@link J2ObjcSrcsProvider} containing source information from the transitive closure.
   */
  public static J2ObjcSrcsProvider j2ObjcSrcsProvider(RuleContext ruleContext) {
    return j2ObjcSrcsProvider(ruleContext, Optional.<J2ObjcSource>absent());
  }

  private static J2ObjcSrcsProvider j2ObjcSrcsProvider(RuleContext ruleContext,
      Optional<J2ObjcSource> currentSource) {
    NestedSetBuilder<J2ObjcSource> builder = NestedSetBuilder.stableOrder();
    builder.addAll(currentSource.asSet());
    boolean hasProtos = currentSource.isPresent()
        && currentSource.get().getSourceType() == J2ObjcSource.SourceType.PROTO;

    for (J2ObjcSrcsProvider provider :
        ruleContext.getPrerequisites("deps", Mode.TARGET, J2ObjcSrcsProvider.class)) {
      builder.addTransitive(provider.getSrcs());
      hasProtos |= provider.hasProtos();
    }

    return new J2ObjcSrcsProvider(builder.build(), hasProtos);
  }

  public static Artifact artifactByAppendingToBaseName(RuleContext context, String suffix) {
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

  public static ObjcConfiguration objcConfiguration(RuleContext ruleContext) {
    return ruleContext.getFragment(ObjcConfiguration.class);
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
   * Attributes for {@code objc_*} rules that can link in SDK frameworks.
   */
  @BlazeRule(name = "$objc_sdk_frameworks_rule",
      type = RuleClassType.ABSTRACT)
  public static class ObjcSdkFrameworksRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          /* <!-- #BLAZE_RULE($objc_sdk_frameworks_rule).ATTRIBUTE(sdk_frameworks) -->
          Names of SDK frameworks to link with. For instance, "XCTest" or
          "Cocoa". "UIKit" and "Foundation" are always included and do not mean
          anything if you include them.
          When linking a library, only those frameworks named in that library's
          sdk_frameworks attribute are linked in. When linking a binary, all
          SDK frameworks named in that binary's transitive dependency graph are
          used.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("sdk_frameworks", STRING_LIST))
          /* <!-- #BLAZE_RULE($objc_sdk_frameworks_rule).ATTRIBUTE(weak_sdk_frameworks) -->
          Names of SDK frameworks to weakly link with. For instance,
          "MediaAccessibility". In difference to regularly linked SDK
          frameworks, symbols from weakly linked frameworks do not cause an
          error if they are not present at runtime.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("weak_sdk_frameworks", STRING_LIST))
          /* <!-- #BLAZE_RULE($objc_sdk_frameworks_rule).ATTRIBUTE(sdk_dylibs) -->
          Names of SDK .dylib libraries to link with. For instance, "libz" or
          "libarchive". "libc++" is included automatically if the binary has
          any C++ or Objective-C++ sources in its dependency tree. When linking
          a binary, all libraries named in that binary's transitive dependency
          graph are used.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("sdk_dylibs", STRING_LIST))
          .build();
    }
  }

  /**
   * Iff a file matches this type, it is considered to use C++.
   */
  static final FileType CPP_SOURCES = FileType.of(".cc", ".cpp", ".mm", ".cxx", ".C");

  private static final FileType NON_CPP_SOURCES = FileType.of(".m", ".c");

  static final FileTypeSet SRCS_TYPE = FileTypeSet.of(NON_CPP_SOURCES, CPP_SOURCES);

  static final FileTypeSet NON_ARC_SRCS_TYPE = FileTypeSet.of(FileType.of(".m", ".mm"));

  static final FileTypeSet PLIST_TYPE = FileTypeSet.of(FileType.of(".plist"));

  static final FileTypeSet STORYBOARD_TYPE = FileTypeSet.of(FileType.of(".storyboard"));

  static final FileType XIB_TYPE = FileType.of(".xib");

  /**
   * Common attributes for {@code objc_*} rules that allow the definition of resources such as
   * storyboards.
   */
  @BlazeRule(name = "$objc_base_resources_rule",
      type = RuleClassType.ABSTRACT,
      ancestors = { BaseRule.class })
  public static class ObjcBaseResourcesRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($objc_base_resources_rule).ATTRIBUTE(strings) -->
          Files which are plists of strings, often localizable. These files
          are converted to binary plists (if they are not already) and placed
          in the bundle root of the final package. If this file's immediate
          containing directory is named *.lproj (e.g. en.lproj, Base.lproj), it
          will be placed under a directory of that name in the final bundle.
          This allows for localizable strings.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("strings", LABEL_LIST).legacyAllowAnyFileType()
              .direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_resources_rule).ATTRIBUTE(xibs) -->
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
          /* <!-- #BLAZE_RULE($objc_base_resources_rule).ATTRIBUTE(storyboards) -->
          Files which are .storyboard resources, possibly localizable. These
          files are compiled to .storyboardc directories, which are placed in
          the bundle root of the final package. If the storyboards's immediate
          containing directory is named *.lproj (e.g. en.lproj, Base.lproj), it
          will be placed under a directory of that name in the final bundle.
          This allows for localizable UI.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("storyboards", LABEL_LIST)
              .allowedFileTypes(STORYBOARD_TYPE))
          /* <!-- #BLAZE_RULE($objc_base_resources_rule).ATTRIBUTE(resources) -->
          Files to include in the final application bundle. They are not
          processed or compiled in any way besides the processing done by the
          rules that actually generate them. These files are placed in the root
          of the bundle (e.g. Payload/foo.app/...) in most cases. However, if
          they appear to be localized (i.e. are contained in a directory called
          *.lproj), they will be placed in a directory of the same name in the
          app bundle.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("resources", LABEL_LIST).legacyAllowAnyFileType().direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_resources_rule).ATTRIBUTE(datamodels) -->
          Files that comprise the data models of the final linked binary.
          Each file must have a containing directory named *.xcdatamodel, which
          is usually contained by another *.xcdatamodeld (note the added d)
          directory.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("datamodels", LABEL_LIST).legacyAllowAnyFileType()
              .direct_compile_time_input())
          /* <!-- #BLAZE_RULE($objc_base_resources_rule).ATTRIBUTE(asset_catalogs) -->
          Files that comprise the asset catalogs of the final linked binary.
          Each file must have a containing directory named *.xcassets. This
          containing directory becomes the root of one of the asset catalogs
          linked with any binary that depends directly or indirectly on this
          target.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("asset_catalogs", LABEL_LIST).legacyAllowAnyFileType()
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
   * Common attributes for {@code objc_*} rules that contain compilable content.
   */
  @BlazeRule(name = "$objc_compilation_rule",
      type = RuleClassType.ABSTRACT,
      ancestors = { BaseRuleClasses.RuleBase.class, ObjcSdkFrameworksRule.class,
                    ObjcBaseResourcesRule.class })
  public static class ObjcCompilationRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($objc_compilation_rule).ATTRIBUTE(hdrs) -->
          The list of Objective-C files that are included as headers by source
          files in this rule or by users of this library.
          ${SYNOPSIS}
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("hdrs", LABEL_LIST)
              .direct_compile_time_input()
              .allowedFileTypes(FileTypeSet.ANY_FILE))
          /* <!-- #BLAZE_RULE($objc_compilation_rule).ATTRIBUTE(includes) -->
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
          /* <!-- #BLAZE_RULE($objc_compilation_rule).ATTRIBUTE(sdk_includes) -->
          List of <code>#include/#import</code> search paths to add to this target
          and all depending targets, where each path is relative to
          <code>$(SDKROOT)/usr/include</code>.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("sdk_includes", Type.STRING_LIST))
          .build();
    }
  }

  /**
   * Common attributes for rules that uses ObjC proto compiler.
   */
  @BlazeRule(name = "$objc_proto_rule",
      type = RuleClassType.ABSTRACT)
  public static class ObjcProtoRule implements RuleDefinition {

    /**
     * A Predicate that returns true if the ObjC proto compiler and its support deps are needed by
     * the current rule.
     *
     * <p>For proto_library rules, this will return true if they have a j2objc_api_version
     * attribute, and it is greater than 0. For other rules, this will return true by default.
     */
    public static final Predicate<AttributeMap> USE_PROTO_COMPILER = new Predicate<AttributeMap>() {
      @Override
      public boolean apply(AttributeMap rule) {
        return rule.getAttributeDefinition("j2objc_api_version") == null
            || rule.get("j2objc_api_version", Type.INTEGER) != 0;
      }
    };

    public static final String COMPILE_PROTOS_ATTR = "$googlemac_proto_compiler";
    public static final String PROTO_SUPPORT_ATTR = "$googlemac_proto_compiler_support";

    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr(COMPILE_PROTOS_ATTR, LABEL)
              .allowedFileTypes(FileType.of(".py"))
              .cfg(HOST)
              .singleArtifact()
              .condition(USE_PROTO_COMPILER)
              .value(env.getLabel("//tools/objc:compile_protos")))
          .add(attr(PROTO_SUPPORT_ATTR, LABEL)
              .legacyAllowAnyFileType()
              .cfg(HOST)
              .condition(USE_PROTO_COMPILER)
              .value(env.getLabel("//tools/objc:proto_support")))
          .build();
    }
  }

  /**
   * Base rule definition for iOS test rules.
   */
  @BlazeRule(name = "$ios_test_base_rule",
      type = RuleClassType.ABSTRACT,
      ancestors = { ObjcBinaryRule.class })
  public static class IosTestBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($ios_test_base_rule).ATTRIBUTE(target_device) -->
          The device against which to run the test.
          ${SYNOPSIS}
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr(IosTest.TARGET_DEVICE, LABEL)
              .allowedFileTypes()
              .allowedRuleClasses("ios_device"))
          /* <!-- #BLAZE_RULE($ios_test_base_rule).ATTRIBUTE(xctest) -->
          Whether this target contains tests using the XCTest testing framework.
          ${SYNOPSIS}
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr(IosTest.IS_XCTEST, BOOLEAN))
          /* <!-- #BLAZE_RULE($ios_test_base_rule).ATTRIBUTE(xctest_app) -->
          A <code>objc_binary</code> target that contains the app bundle to test against in XCTest.
          This attribute is only valid if <code>xctest</code> is true.
          ${SYNOPSIS}
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr(IosTest.XCTEST_APP, LABEL)
              .value(new Attribute.ComputedDefault(IosTest.IS_XCTEST) {
                @Override
                public Object getDefault(AttributeMap rule) {
                  return rule.get(IosTest.IS_XCTEST, Type.BOOLEAN)
                      ? env.getLabel("//tools/objc:xctest_app")
                      : null;
                }
              })
              .allowedFileTypes()
              .allowedRuleClasses("objc_binary"))
          .override(attr("infoplist", LABEL)
              .value(new Attribute.ComputedDefault(IosTest.IS_XCTEST) {
                @Override
                public Object getDefault(AttributeMap rule) {
                  return rule.get(IosTest.IS_XCTEST, Type.BOOLEAN)
                      ? env.getLabel("//tools/objc:xctest_infoplist")
                      : null;
                }
              })
              .allowedFileTypes(PLIST_TYPE))
          .build();
    }
  }

  /**
   * Abstract rule type with the {@code infoplist} attribute.
   */
  @BlazeRule(name = "$objc_has_infoplist_rule",
      type = RuleClassType.ABSTRACT)
  public static class ObjcHasInfoplistRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($objc_has_infoplist_rule).ATTRIBUTE(infoplist) -->
          The infoplist file. This corresponds to <i>appname</i>-Info.plist in Xcode projects.
          ${SYNOPSIS}
          Blaze will perform variable substitution on the plist file for the following values:
          <ul>
            <li><code>${EXECUTABLE_NAME}</code>: The name of the executable generated and included
               in the bundle by blaze, which can be used as the value for
               <code>CFBundleExecutable</code> within the plist.
            <li><code>${BUNDLE_NAME}</code>: This target's name and bundle suffix (.bundle or .app)
               in the form<code><var>name</var></code>.<code>suffix</code>.
            <li><code>${PRODUCT_NAME}</code>: This target's name.
         </ul>
         <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("infoplist", LABEL)
            .allowedFileTypes(PLIST_TYPE))
        .build();
    }
  }

  /**
   * Abstract rule type with the {@code entitlements} attribute.
   */
  @BlazeRule(name = "$objc_has_entitlements_rule",
      type = RuleClassType.ABSTRACT)
  public static class ObjcHasEntitlementsRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($objc_has_entitlements_rule).ATTRIBUTE(entitlements) -->
          The entitlements file required for device builds of this application. See
          <a href="https://developer.apple.com/library/mac/documentation/Miscellaneous/Reference/EntitlementKeyReference/Chapters/AboutEntitlements.html">the apple documentation</a>
          for more information. If absent, the default entitlements from the
          provisioning profile will be used.
          <p>
          The following variables are substituted as per
          <a href="https://developer.apple.com/library/ios/documentation/General/Reference/InfoPlistKeyReference/Articles/CoreFoundationKeys.html">their definitions in Apple's documentation</a>:
          $(AppIdentifierPrefix) and $(CFBundleIdentifier).
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("entitlements", LABEL).legacyAllowAnyFileType())
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

