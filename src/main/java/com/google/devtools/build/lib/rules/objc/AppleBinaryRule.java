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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_KEYED_STRING_DICT;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagProvider;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagTransitionFactory;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;

/**
 * Rule definition for apple_binary.
 */
public class AppleBinaryRule implements RuleDefinition {

  public static final String BINARY_TYPE_ATTR = "binary_type";
  public static final String BUNDLE_LOADER_ATTR_NAME = "bundle_loader";
  public static final String EXTENSION_SAFE_ATTR_NAME = "extension_safe";

  private final ObjcProtoAspect objcProtoAspect;

  /**
   * Constructor that returns a newly configured AppleBinaryRule object.
   */
  public AppleBinaryRule(ObjcProtoAspect objcProtoAspect) {
    this.objcProtoAspect = objcProtoAspect;
  }

  /**
   * There are 3 classes of fully linked binaries in Mach: executable, dynamic library, and loadable
   * bundle.
   *
   * <p>The executable is the binary that can be run directly by the operating system. It implements
   * implements the main method that is the entry point to the program. In Apple apps, they are
   * usually distributed in .app bundles, which are folders that contain the executable along with
   * required resources to run.
   *
   * <p>Dynamic libraries are binaries meant to be loaded at load time (when the operating system is
   * loading the binary into memory), and they _cant'_ be unloaded. This is a great way to reduce
   * binary size of executables by providing a dynamic library that groups common functionality into
   * one dynamic library that can then be loaded by multiple executables. They are usually
   * distributed in frameworks, which are .framework bundles that contain the dylib as well as well
   * as required resources to run.
   *
   * <p>Loadable bundles are binaries that can be loaded by other binaries at runtime, and they
   * can't be directly executed by the operating system. When linking, a bundle_loader binary may be
   * passed which signals the linker on where to look for unimplemented symbols, basically declaring
   * that the bundle should be loaded by that binary. Bundle binaries are usually found in Plugins,
   * and one common use case is tests. Tests are bundled into an .xctest bundle which contains the
   * tests binary along with required resources. The test bundle is then loaded and run during test
   * execution.
   *
   * <p>The binary type is configurable via the "binary_type" attribute described below.
   */
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(
            ObjcConfiguration.class,
            J2ObjcConfiguration.class,
            AppleConfiguration.class,
            CppConfiguration.class)
        .add(
            attr("$is_executable", BOOLEAN)
                .value(true)
                .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target"))
        /* <!-- #BLAZE_RULE(apple_binary).ATTRIBUTE(binary_type) -->
        The type of binary that this target should build.

        Options are:
        <ul>
          <li>
            <code>executable</code> (default): the output binary is an executable and must implement
            the main() function.
          </li><li>
            <code>loadable_bundle</code>: the output binary is a loadable bundle that may be loaded
            at runtime. When building a bundle, you may also pass a bundle_loader binary that
            contains symbols referenced but not implemented in the loadable bundle.
          </li><li>
            <code>dylib</code>: the output binary is meant to be loaded at load time (when the
            operating system is loading the binary into memory) and cannot be unloaded. Dylibs
            are usually consumed in frameworks, which are .framework bundles that contain the
            dylib as well as well as required resources to run.
          </li>
        </ul>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr(BINARY_TYPE_ATTR, STRING)
                .value(AppleBinary.BinaryType.EXECUTABLE.toString())
                .allowedValues(new AllowedValueSet(AppleBinary.BinaryType.getValues())))
        /* <!-- #BLAZE_RULE(apple_binary).ATTRIBUTE(extension_safe) -->
        This attribute is deprecated and will be removed soon. It currently has no effect.
        "Extension-safe" link options may be added using the <code>linkopts</code> attribute.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(EXTENSION_SAFE_ATTR_NAME, BOOLEAN)
                .value(false)
                .nonconfigurable("Determines the configuration transition on deps"))
        .add(
            attr(BUNDLE_LOADER_ATTR_NAME, LABEL)
                .direct_compile_time_input()
                .mandatoryProviders(
                    ImmutableList.of(
                        SkylarkProviderIdentifier.forKey(
                            AppleExecutableBinaryInfo.SKYLARK_CONSTRUCTOR.getKey())))
                .allowedFileTypes()
                .singleArtifact()
                .aspect(objcProtoAspect))
        .add(
            attr("feature_flags", LABEL_KEYED_STRING_DICT)
                .undocumented("the feature flag feature has not yet been launched")
                .allowedRuleClasses("config_feature_flag")
                .allowedFileTypes()
                .nonconfigurable("defines an aspect of configuration")
                .mandatoryProviders(ImmutableList.of(ConfigFeatureFlagProvider.id())))
        /*<!-- #BLAZE_RULE(apple_binary).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>_lipobin</code>: the 'lipo'ed potentially multi-architecture
             binary. All transitive dependencies and <code>srcs</code> are linked.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(ObjcRuleClasses.LIPOBIN_OUTPUT))
        .cfg(
            ComposingTransitionFactory.of(
                (rule) -> AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION,
                new ConfigFeatureFlagTransitionFactory("feature_flags")))
        .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(env))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_binary")
        .factoryClass(AppleBinary.class)
        .ancestors(BaseRuleClasses.BaseRule.class, ObjcRuleClasses.MultiArchPlatformRule.class,
            ObjcRuleClasses.DylibDependingRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = apple_binary, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule produces single- or multi-architecture ("fat") Objective-C libraries and/or binaries,
typically used in creating apple bundles, such as frameworks, extensions, or applications.</p>

<p>The <code>lipo</code> tool is used to combine files of multiple architectures. One of several
flags may control which architectures are included in the output, depending on the value of
the "platform_type" attribute.</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
