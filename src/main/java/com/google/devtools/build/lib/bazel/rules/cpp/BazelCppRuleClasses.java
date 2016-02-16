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

package com.google.devtools.build.lib.bazel.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST_DICT;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ALWAYS_LINK_LIBRARY;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ALWAYS_LINK_PIC_LIBRARY;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ARCHIVE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ASSEMBLER;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.CPP_HEADER;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.CPP_SOURCE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.C_SOURCE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.OBJECT_FILE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.PIC_ARCHIVE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.PIC_OBJECT_FILE;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.SHARED_LIBRARY;
import static com.google.devtools.build.lib.rules.cpp.CppFileTypes.VERSIONED_SHARED_LIBRARY;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.base.Predicates;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.bazel.rules.BazelBaseRuleClasses;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain.RequiresXcodeConfigRule;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;

/**
 * Rule class definitions for C++ rules.
 */
public class BazelCppRuleClasses {
  static final SafeImplicitOutputsFunction CC_LIBRARY_DYNAMIC_LIB =
      fromTemplates("%{dirname}lib%{basename}.so");

  static final SafeImplicitOutputsFunction CC_BINARY_IMPLICIT_OUTPUTS =
      fromFunctions(CppRuleClasses.CC_BINARY_STRIPPED, CppRuleClasses.CC_BINARY_DEBUG_PACKAGE);

  static final FileTypeSet ALLOWED_SRC_FILES = FileTypeSet.of(
      CPP_SOURCE,
      C_SOURCE,
      CPP_HEADER,
      ASSEMBLER_WITH_C_PREPROCESSOR,
      ASSEMBLER,
      ARCHIVE,
      PIC_ARCHIVE,
      ALWAYS_LINK_LIBRARY,
      ALWAYS_LINK_PIC_LIBRARY,
      SHARED_LIBRARY,
      VERSIONED_SHARED_LIBRARY,
      OBJECT_FILE,
      PIC_OBJECT_FILE);

  static final String[] DEPS_ALLOWED_RULES = new String[] {
      "cc_library",
  };

  /**
   * Miscellaneous configuration transitions. It would be better not to have this - please don't add
   * to it.
   */
  public static enum CppTransition implements Transition {
    /**
     * The configuration for LIPO information collection. Requesting this from a configuration that
     * does not have lipo optimization enabled may result in an exception.
     */
    LIPO_COLLECTOR,

    /**
     * The corresponding (target) configuration.
     */
    TARGET_CONFIG_FOR_LIPO;

    @Override
    public boolean defaultsToSelf() {
      return false;
    }
  }

  private static final RuleClass.Configurator<BuildConfiguration, Rule> LIPO_ON_DEMAND =
      new RuleClass.Configurator<BuildConfiguration, Rule>() {
    @Override
    public BuildConfiguration apply(Rule rule, BuildConfiguration configuration) {
      if (configuration.useDynamicConfigurations()) {
        // Dynamic configurations don't currently work with LIPO. partially because of lack of
        // support for TARGET_CONFIG_FOR_LIPO. We can't check for LIPO here because we have
        // to apply TARGET_CONFIG_FOR_LIPO to determine it, So we just assume LIPO is disabled.
        // This is safe because Bazel errors out if the two options are combined.
        return configuration;
      }
      BuildConfiguration toplevelConfig =
          configuration.getConfiguration(CppTransition.TARGET_CONFIG_FOR_LIPO);
      // If LIPO is enabled, override the default configuration.
      if (toplevelConfig != null
          && toplevelConfig.getFragment(CppConfiguration.class).isLipoOptimization()
          && !configuration.isHostConfiguration()
          && !configuration.getFragment(CppConfiguration.class).isLipoContextCollector()) {
        // Switch back to data when the cc_binary is not the LIPO context.
        return (rule.getLabel().equals(
            toplevelConfig.getFragment(CppConfiguration.class).getLipoContextLabel()))
            ? toplevelConfig
            : configuration.getTransitions().getStaticConfiguration(ConfigurationTransition.DATA);
      }
      return configuration;
    }

    @Override
    public String getCategory() {
      return "lipo";
    }
  };

  /**
   * Label of a pseudo-filegroup that contains all crosstool and libcfiles for
   * all configurations, as specified on the command-line.
   */
  public static final String CROSSTOOL_LABEL = "//tools/defaults:crosstool";

  public static final LateBoundLabel<BuildConfiguration> CC_TOOLCHAIN =
      new LateBoundLabel<BuildConfiguration>(CROSSTOOL_LABEL) {
        @Override
        public Label getDefault(Rule rule, AttributeMap attributes,
            BuildConfiguration configuration) {
          return configuration.getFragment(CppConfiguration.class).getCcToolchainRuleLabel();
        }
      };

  public static final LateBoundLabel<BuildConfiguration> DEFAULT_MALLOC =
      new LateBoundLabel<BuildConfiguration>() {
        @Override
        public Label getDefault(Rule rule, AttributeMap attributes,
            BuildConfiguration configuration) {
          return configuration.getFragment(CppConfiguration.class).customMalloc();
        }
      };

  public static final LateBoundLabel<BuildConfiguration> STL =
      new LateBoundLabel<BuildConfiguration>() {
        @Override
        public Label getDefault(Rule rule, AttributeMap attributes,
            BuildConfiguration configuration) {
          return getStl(rule, configuration);
        }
      };

  /**
   * Implementation for the :lipo_context_collector attribute.
   */
  public static final LateBoundLabel<BuildConfiguration> LIPO_CONTEXT_COLLECTOR =
      new LateBoundLabel<BuildConfiguration>() {
    @Override
    public Label getDefault(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
      // This attribute connects a target to the LIPO context target configured with the
      // lipo input collector configuration.
      CppConfiguration cppConfiguration = configuration.getFragment(CppConfiguration.class);
      return !cppConfiguration.isLipoContextCollector()
          && (cppConfiguration.getLipoMode() == LipoMode.BINARY)
          ? cppConfiguration.getLipoContextLabel()
          : null;
    }
  };

  /**
   * Returns the STL prerequisite of the rule.
   *
   * <p>If rule has an implicit $stl_default attribute returns STL version set on the
   * command line or if not set, the value of the $stl_default attribute. Returns
   * {@code null} otherwise.
   */
  private static Label getStl(Rule rule, BuildConfiguration original) {
    Label stl = null;
    if (rule.getRuleClassObject().hasAttr("$stl_default", BuildType.LABEL)) {
      Label stlConfigLabel = original.getFragment(CppConfiguration.class).getStl();
      Label stlRuleLabel = RawAttributeMapper.of(rule).get("$stl_default", BuildType.LABEL);
      if (stlConfigLabel == null) {
        stl = stlRuleLabel;
      } else if (!stlConfigLabel.equals(rule.getLabel()) && stlRuleLabel != null) {
        // prevents self-reference and a cycle through standard STL in the dependency graph
        stl = stlConfigLabel;
      }
    }
    return stl;
  }

  /**
   * Common attributes for all rules that create C++ links. This may
   * include non-cc_* rules (e.g. py_binary).
   */
  public static final class CcLinkingRule implements RuleDefinition {
    @Override
    @SuppressWarnings("unchecked")
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr(":cc_toolchain", LABEL).value(CC_TOOLCHAIN))
          .setPreferredDependencyPredicate(Predicates.<String>or(CPP_SOURCE, C_SOURCE, CPP_HEADER))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_linking_rule")
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }

  /**
   * Common attributes for C++ rules.
   */
  public static final class CcBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /*<!-- #BLAZE_RULE($cc_base_rule).ATTRIBUTE(copts) -->
          Add these options to the C++ compilation command.
          Subject to <a href="make-variables.html">"Make variable"</a> substitution and
          <a href="common-definitions.html#sh-tokenization">
          Bourne shell tokenization</a>.
          <p>Each string in this attribute is added in the given order to <code>COPTS</code>
          before compiling the binary target.
          The flags take effect only for compiling this target, not its dependencies,
          so be careful about header files included elsewhere.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("copts", STRING_LIST))
          .add(
              attr("$stl_default", LABEL)
                  .value(env.getToolsLabel("//tools/cpp:stl")))
          .add(attr(":stl", LABEL).value(STL))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_base_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(CcLinkingRule.class)
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  public static final class CcDeclRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /*<!-- #BLAZE_RULE($cc_decl_rule).ATTRIBUTE(abi)[DEPRECATED] -->
           Platform-specific information string which is used in combination
            with <code>abi_deps</code>.
            Subject to <a href="make-variables.html">"Make" variable</a> substitution.
            <p>
              This string typically includes references to one or more "Make" variables of the form
              <code>"$(VAR)"</code>. The default value is <code>"$(ABI)"</code>.
            </p>
            <p>
              With <code>abi_deps</code>, the regular expression <code>patterns</code> will be
              matched against this string.
            </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          // Deprecated in favor of configurable attributes.
          .add(attr("abi", STRING).value("$(ABI)"))
          /*<!-- #BLAZE_RULE($cc_decl_rule).ATTRIBUTE(abi_deps)[DEPRECATED] -->
          The list of architecture-specific dependencies that are processed to create the target.
          <i>(Dictionary mapping strings to lists of
             <a href="../build-ref.html#labels">labels</a>; optional)</i>
          <p><i><a href="common-definitions.html#configurable-attributes">
            Configurable attributes</a> is a generalization
            of the same concept that works for most rules and attributes. It deprecates
            <code>abi_deps</code>, which we intend to ultimately remove. Use configurable
            attributes over <code>abi_deps</code> whenever possible. When not possible, let
            us know why.</i>
          </p>
          <p>Each entry in this dictionary follows the form of
             <code>'pattern' : ['label1', 'label2', ...]</code>.  If the library's
             <code>abi</code> attribute is an unanchored match for the regular
             expression defined in <code>pattern</code>, the corresponding
             labels are used as dependencies as if they had appeared in the
             <code>deps</code> list.
          </p>
          <p>All pairs with a matching <code>pattern</code> will have their labels
             used.  If no matches are found, no dependencies will be used.  The
             ordering is irrelevant.
          </p>
          <p>If you want a <code>pattern</code> to not match a particular
             <code>abi</code>, for example adding a dep on all non-k8 platforms, you
             can use a negative lookahead pattern.  This would look like
             <code>(?!k8).*</code>.
          </p>
          <p>If using <code>abi_deps</code>, do not provide <code>deps</code>.
             Instead, use an entry with a <code>pattern</code> of <code>'.*'</code>
             because that matches everything.  This is also how to share
             dependencies across multiple different <code>abi</code> values.
          </p>
          <p>Typically, this mechanism is used to specify the appropriate set of
             paths to pre-compiled libraries for the target architecture of the
             current build.  Such paths are parameterized over "Make" variables
             such as <code>$(ABI)</code>, <code>$(TARGET_CPU)</code>,
             <code>$(C_COMPILER)</code>, etc, but since "Make" variables are not
             allowed in <a href="../build-ref.html#labels">labels</a>, the
             architecture-specific files cannot be specified via the normal
             <code>srcs</code> attribute. Instead, this mechanism can be used
             to declare architecture-specific dependent rules for the current
             target that can specify the correct libraries in their own
             <code>srcs</code>.
          </p>
          <p>This mechanism is also used to specify the appropriate set of
             dependencies when some targets can't compile for the target architecture
             of the current build.  In most cases, uses an <code>#ifdef</code>.
             Only use <code>abi_deps</code> for more significant dependency changes.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          // Deprecated in favor of configurable attributes.
          .add(attr("abi_deps", LABEL_LIST_DICT))
          /*<!-- #BLAZE_RULE($cc_decl_rule).ATTRIBUTE(defines) -->
          List of defines to add to the compile line.
          Subject to <a href="make-variables.html">"Make" variable</a> substitution and
          <a href="common-definitions.html#sh-tokenization">
          Bourne shell tokenization</a>.
          Each string, which must consist of a single Bourne shell token,
          is prepended with <code>-D</code> and added to
          <code>COPTS</code>.
          Unlike <a href="#cc_binary.copts"><code>copts</code></a>, these flags are added for the
          target and every rule that depends on it!  Be very careful, since this may have
          far-reaching effects.  When in doubt, add "-D" flags to
          <a href="#cc_binary.copts"><code>copts</code></a> instead.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("defines", STRING_LIST))
          /*<!-- #BLAZE_RULE($cc_decl_rule).ATTRIBUTE(includes) -->
          List of include dirs to be added to the compile line.
          <p>Subject to <a href="make-variables.html">"Make variable"</a> substitution.
             Each string is prepended with <code>-isystem</code> and added to <code>COPTS</code>.
             Unlike <a href="#cc_binary.copts">COPTS</a>, these flags are added for this rule
             and every rule that depends on it. (Note: not the rules it depends upon!) Be
             very careful, since this may have far-reaching effects.  When in doubt, add
             "-I" flags to <a href="#cc_binary.copts">COPTS</a> instead.
          </p>
          <p>To use <code>-iquote</code> instead of <code>-isystem</code>, specify
             <code>--use_isystem_for_includes=false</code> (the flag is undocumented and defaults
             to <code>true</code>).
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("includes", STRING_LIST))
          .add(attr(":lipo_context_collector", LABEL)
              .cfg(CppTransition.LIPO_COLLECTOR)
              .value(LIPO_CONTEXT_COLLECTOR)
              .skipPrereqValidatorCheck())
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_decl_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(BaseRuleClasses.RuleBase.class)
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  public static final class CcRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          /*<!-- #BLAZE_RULE($cc_rule).ATTRIBUTE(srcs) -->
          The list of C and C++ files that are processed to create the target.
          These are C/C++ source and header files, either non-generated (normal source
          code) or generated.
          <p>All <code>.cc</code>, <code>.c</code>, and <code>.cpp</code> files will
             be compiled. These might be generated files: if a named file is in
             the <code>outs</code> of some other rule, this rule
             will automatically depend on that other rule.
          </p>
          <p>A <code>.h</code> file will not be compiled, but will be available for
             inclusion by sources in this rule. Both <code>.cc</code> and
             <code>.h</code> files can directly include headers listed in
             these <code>srcs</code> or in the <code>hdrs</code> of any rule listed in
             the <code>deps</code> argument.
          </p>
          <p>All <code>#include</code>d files must be mentioned in the
             <code>srcs</code> attribute of this rule, or in the
             <code>hdrs</code> attribute of referenced <code>cc_library()</code>s.
             The recommended style is for headers associated with a library to be
             listed in that library's <code>hdrs</code> attribute, and any remaining
             headers associated with this rule's sources to be listed in
             <code>srcs</code>. See <a href="#hdrs">"Header inclusion checking"</a>
             for a more detailed description.
          </p>
          <p>If a rule's name is in the <code>srcs</code>,
             then this rule automatically depends on that one.
             If the named rule's <code>outs</code> are C or C++
             source files, they are compiled into this rule;
             if they are library files, they are linked in.
          </p>
          <p>
            Permitted <code>srcs</code> file types:
          </p>
          <ul>
            <li>C and C++ source files: <code>.c</code>, <code>.cc</code>, <code>.cpp</code>,
              <code>.cxx</code>, <code>.c++</code, <code>.C</code></li>
            <li>C and C++ header files: <code>.h</code>, <code>.hh</code>, <code>.hpp</code>,
              <code>.hxx</code>, <code>.inc</code></li>
            <li>Assembler with C preprocessor: <code>.S</code></li>
            <li>Archive: <code>.a</code>, <code>.pic.a</code></li>
            <li>"Always link" library: <code>.lo</code>, <code>.pic.lo</code></li>
            <li>Shared library, versioned or unversioned: <code>.so</code>,
              <code>.so.<i>version</i></code></li>
            <li>Object file: <code>.o</code>, <code>.pic.o</code></li>
          </ul>
          <p>
            ...and any rules that produce those files.
            Different extensions denote different programming languages in
            accordance with gcc convention.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(
              attr("srcs", LABEL_LIST)
                  .direct_compile_time_input()
                  .allowedFileTypes(ALLOWED_SRC_FILES))
          /*<!-- #BLAZE_RULE($cc_rule).ATTRIBUTE(deps) -->
          The list of other libraries to be linked in to the binary target.
          <p>These are always <code>cc_library</code> rules.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .override(
              attr("deps", LABEL_LIST)
                  .allowedRuleClasses(DEPS_ALLOWED_RULES)
                  .allowedFileTypes(CppFileTypes.LINKER_SCRIPT)
                  .skipAnalysisTimeFileTypeCheck())
          /*<!-- #BLAZE_RULE($cc_rule).ATTRIBUTE(linkopts) -->
          Add these flags to the C++ linker command.
          Subject to <a href="make-variables.html">"Make" variable</a> substitution,
          <a href="common-definitions.html#sh-tokenization">
          Bourne shell tokenization</a> and
          <a href="common-definitions.html#label-expansion">label expansion</a>.
          Each string in this attribute is added to <code>LINKOPTS</code> before
          linking the binary target.
          <p>
            Each element of this list that does not start with <code>$</code> or <code>-</code> is
            assumed to be the label of a target in <code>deps</code>.  The
            list of files generated by that target is appended to the linker
            options.  An error is reported if the label is invalid, or is
            not declared in <code>deps</code>.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("linkopts", STRING_LIST))
          /*<!-- #BLAZE_RULE($cc_rule).ATTRIBUTE(nocopts) -->
          Remove matching options from the C++ compilation command.
          Subject to <a href="make-variables.html">"Make" variable</a> substitution.
          The value of this attribute is interpreted as a regular expression.
          Any preexisting <code>COPTS</code> that match this regular expression
          (not including values explicitly specified in the rule's <a
          href="#cc_binary.copts">copts</a> attribute) will be removed from
          <code>COPTS</code> for purposes of compiling this rule.
          This attribute should rarely be needed.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("nocopts", STRING))
          /*<!-- #BLAZE_RULE($cc_rule).ATTRIBUTE(linkstatic) -->
           Link the binary in mostly-static mode.
           By default this option is on for <code>cc_binary</code> and off for <code>cc_test</code>.
           <p>
             If enabled, this tells the build tool to link in <code>.a</code>'s instead of
             <code>.so</code>'s for user libraries whenever possible. Some system libraries may
             still be linked dynamically, as are libraries for which there's no static library. So
             the resulting binary will be dynamically linked, hence only <i>mostly</i> static.
           </p>
           <p>There are really three different ways to link an executable:</p>
           <ul>
           <li> FULLY STATIC, in which everything is linked statically; e.g. "<code>gcc -static
             foo.o libbar.a libbaz.a -lm</code>".<br/>This mode is enabled by specifying
             <code>-static</code> in the <a href="#cc_binary.linkopts"><code>linkopts</code></a>
             attribute.</li>
           <li> MOSTLY STATIC, in which all user libraries are linked statically (if a static
             version is available), but where system libraries are linked dynamically, e.g.
             "<code>gcc foo.o libfoo.a libbaz.a -lm</code>".<br/>This mode is enabled by specifying
             <code>linkstatic=1</code>.</li>
           <li> DYNAMIC, in which all libraries are linked dynamically (if a dynamic version is
             available), e.g. "<code>gcc foo.o libfoo.so libbaz.so -lm</code>".<br/> This mode is
             enabled by specifying <code>linkstatic=0</code>.</li>
           </ul>
           <p>
           The <code>linkstatic</code> attribute has a different meaning if used on a
           <a href="#cc_library"><code>cc_library()</code></a> rule.
           For a C++ library, <code>linkstatic=1</code> indicates that only
           static linking is allowed, so no <code>.so</code> will be produced.
           </p>
           <p>
           If <code>linkstatic=0</code>, then the build tool will create symlinks to
           depended-upon shared libraries in the <code>*.runfiles</code> area.
           </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("linkstatic", BOOLEAN).value(true))
          .override(
              attr("$stl_default", LABEL)
                  .value(
                      new Attribute.ComputedDefault() {
                        @Override
                        public Object getDefault(AttributeMap rule) {
                          // Every cc_rule depends implicitly on STL to make
                          // sure that the correct headers are used for inclusion.
                          // The only exception is STL itself,
                          // to avoid cycles in the dependency graph.
                          Label stl = env.getToolsLabel("//tools/cpp:stl");
                          return rule.getLabel().equals(stl) ? null : stl;
                        }
                      }))
          .build();
    }
    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(CcDeclRule.class, CcBaseRule.class)
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  public static final class CcBinaryBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /*<!-- #BLAZE_RULE($cc_binary_base).ATTRIBUTE(malloc) -->
          Override the default dependency on malloc.
          <p>
            By default, Linux C++ binaries are linked against <code>//tools/cpp:malloc</code>,
            which is an empty library so the binary ends up using libc malloc. This label must
            refer to a <code>cc_library</code>. If compilation is for a non-C++ rule, this option
            has no effect. The value of this attribute is ignored if <code>linkshared=1</code> is
            specified.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("malloc", LABEL)
              .value(env.getToolsLabel("//tools/cpp:malloc"))
              .allowedFileTypes()
              .allowedRuleClasses("cc_library"))
          .add(attr(":default_malloc", LABEL).value(DEFAULT_MALLOC))
          /*<!-- #BLAZE_RULE($cc_binary_base).ATTRIBUTE(stamp) -->
          Enable link stamping.
          Whether to encode build information into the binary. Possible values:
          <ul>
            <li><code>stamp = 1</code>: Stamp the build information into the
              binary. Stamped binaries are only rebuilt when their dependencies
              change. Use this if there are tests that depend on the build
              information.</li>
            <li><code>stamp = 0</code>: Always replace build information by constant
              values. This gives good build result caching.</li>
            <li><code>stamp = -1</code>: Embedding of build information is controlled
              by the <a href="../blaze-user-manual.html#flag--stamp">--[no]stamp</a> flag.</li>
          </ul>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          // TODO(bazel-team): document this. Figure out a standard way to access stamp data at
          // runtime.
          .add(attr("stamp", TRISTATE).value(TriState.AUTO))
          .build();
    }
    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_binary_base")
          .type(RuleClassType.ABSTRACT)
          .ancestors(CcRule.class, RequiresXcodeConfigRule.class)
          .build();
    }
  }

  /**
   * Rule definition for cc_binary rules.
   */
  public static final class CcBinaryRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .requiresConfigurationFragments(CppConfiguration.class, AppleConfiguration.class)
          /*<!-- #BLAZE_RULE(cc_binary).IMPLICIT_OUTPUTS -->
          <ul>
          <li><code><var>name</var>.stripped</code> (only built if explicitly requested): A stripped
            version of the binary. <code>strip -g</code> is run on the binary to remove debug
            symbols.  Additional strip options can be provided on the command line using
            <code>--stripopt=-foo</code>. This output is only built if explicitly requested.</li>
          <li><code><var>name</var>.dwp</code> (only built if explicitly requested): If
            <a href="https://gcc.gnu.org/wiki/DebugFission">Fission</a> is enabled: a debug
            information package file suitable for debugging remotely deployed binaries. Else: an
            empty file.</li>
          </ul>
          <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
          .setImplicitOutputsFunction(CC_BINARY_IMPLICIT_OUTPUTS)
          /*<!-- #BLAZE_RULE(cc_binary).ATTRIBUTE(linkshared) -->
          Create a shared library.
          To enable this attribute, include <code>linkshared=1</code> in your rule. By default
          this option is off. If you enable it, you must name your binary
          <code>lib<i>foo</i>.so</code> (or whatever is the naming convention of libraries on the
          target platform) for some sensible value of <i>foo</i>.
          <p>
            The presence of this flag means that linking occurs with the <code>-shared</code> flag
            to <code>gcc</code>, and the resulting shared library is suitable for loading into for
            example a Java program. However, for build purposes it will never be linked into the
            dependent binary, as it is assumed that shared libraries built with a
            <a href="#cc_binary">cc_binary</a> rule are only loaded manually by other programs, so
            it should not be considered a substitute for the <a href="#cc_library">cc_library</a>
            rule. For sake of scalability we recommend avoiding this approach altogether and
            simply letting <code>java_library</code> depend on <code>cc_library</code> rules
            instead.
          </p>
          <p>
            If you specify both <code>linkopts=['-static']</code> and <code>linkshared=1</code>,
            you get a single completely self-contained unit. If you specify both
            <code>linkstatic=1</code> and <code>linkshared=1</code>, you get a single, mostly
            self-contained unit.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("linkshared", BOOLEAN).value(false)
              .nonconfigurable("used to *determine* the rule's configuration"))
          .cfg(LIPO_ON_DEMAND)
          .build();
    }
    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("cc_binary")
          .ancestors(CcBinaryBaseRule.class, BazelBaseRuleClasses.BinaryBaseRule.class)
          .factoryClass(BazelCcBinary.class)
          .build();
    }
  }

  /**
   * Implementation for the :lipo_context attribute.
   */
  private static final LateBoundLabel<BuildConfiguration> LIPO_CONTEXT =
      new LateBoundLabel<BuildConfiguration>() {
    @Override
    public Label getDefault(Rule rule, AttributeMap attributes, BuildConfiguration configuration) {
      Label result = configuration.getFragment(CppConfiguration.class).getLipoContextLabel();
      return (rule == null || rule.getLabel().equals(result)) ? null : result;
    }
  };

  /**
   * Rule definition for cc_test rules.
   */
  public static final class CcTestRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .requiresConfigurationFragments(CppConfiguration.class)
          .setImplicitOutputsFunction(CppRuleClasses.CC_BINARY_DEBUG_PACKAGE)
          .override(attr("linkstatic", BOOLEAN).value(false))
          .override(attr("stamp", TRISTATE).value(TriState.NO))
          .add(attr(":lipo_context", LABEL).value(LIPO_CONTEXT))
          .build();
    }
    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("cc_test")
          .type(RuleClassType.TEST)
          .ancestors(CcBinaryBaseRule.class, BaseRuleClasses.TestBaseRule.class)
          .factoryClass(BazelCcTest.class)
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  public static final class CcLibraryBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /*<!-- #BLAZE_RULE($cc_library).ATTRIBUTE(hdrs) -->
           The list of header files published by
           this library to be directly included by sources in dependent rules.
          <p>This is the strongly preferred location for declaring header files that
             describe the interface for the library. These headers will be made
             available for inclusion by sources in this rule or in dependent rules.
             Headers not meant to be included by a client of this library should be
             listed in the <code>srcs</code> attribute instead, even if they are
             included by a published header. See <a href="#hdrs">"Header inclusion
             checking"</a> for a more detailed description. </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("hdrs", LABEL_LIST).orderIndependent().direct_compile_time_input()
              .allowedFileTypes(FileTypeSet.ANY_FILE))
          /* <!-- #BLAZE_RULE($cc_library).ATTRIBUTE(textual_hdrs) -->
           The list of header files published by
           this library to be textually included by sources in dependent rules.
          <p>This is the location for declaring header files that cannot be compiled on their own;
             that is, they always need to be textually included by other source files to build valid
             code.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("textual_hdrs", LABEL_LIST).orderIndependent().direct_compile_time_input()
              .legacyAllowAnyFileType())
          // TODO(bazel-team): document or remove.
          .add(attr("linkstamp", LABEL).allowedFileTypes(CPP_SOURCE, C_SOURCE))
          .build();
    }
    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_library")
          .type(RuleClassType.ABSTRACT)
          .ancestors(CcRule.class)
          .build();
    }
  }

  /**
   * Rule definition for the cc_library rule.
   */
  public static final class CcLibraryRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          // TODO: Google cc_library overrides documentation for:
          // deps, data, linkopts, defines, srcs; override here too?

          .requiresConfigurationFragments(CppConfiguration.class, AppleConfiguration.class)
          /*<!-- #BLAZE_RULE(cc_library).ATTRIBUTE(alwayslink) -->
          If 1, any binary that depends (directly or indirectly) on this C++
          library will link in all the object files for the files listed in
          <code>srcs</code>, even if some contain no symbols referenced by the binary.
          This is useful if your code isn't explicitly called by code in
          the binary, e.g., if your code registers to receive some callback
          provided by some service.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("alwayslink", BOOLEAN).
              nonconfigurable("value is referenced in an ImplicitOutputsFunction"))
          .override(attr("linkstatic", BOOLEAN).value(false)
              .nonconfigurable("value is referenced in an ImplicitOutputsFunction"))
          .build();
    }
    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("cc_library")
          .ancestors(CcLibraryBaseRule.class, RequiresXcodeConfigRule.class)
          .factoryClass(BazelCcLibrary.class)
          .build();
    }
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_binary, TYPE = BINARY, FAMILY = C / C++) -->

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/


/*<!-- #BLAZE_RULE (NAME = cc_library, TYPE = LIBRARY, FAMILY = C / C++) -->

<h4 id="hdrs">Header inclusion checking</h4>

<p>
  All header files that are used in the build must be declared in the <code>hdrs</code> or
  <code>srcs</code> of <code>cc_*</code> rules. This is enforced.
</p>

<p>
  For <code>cc_library</code> rules, headers in <code>hdrs</code> comprise the public interface of
  the library and can be directly included both from the files in <code>hdrs</code> and
  <code>srcs</code> of the library itself as well as from files in <code>hdrs</code> and
  <code>srcs</code> of <code>cc_*</code> rules that list the library in their <code>deps</code>.
  Headers in <code>srcs</code> must only be directly included from the files in <code>hdrs</code>
  and <code>srcs</code> of the library itself. When deciding whether to put a header into
  <code>hdrs</code> or <code>srcs</code>, you should ask whether you want consumers of this library
  to be able to directly include it. This is roughly the same decision as between
  <code>public</code> and <code>private</code> visibility in programming languages.
</p>

<p>
  <code>cc_binary</code> and <code>cc_test</code> rules do not have an exported interface, so they
  also do not have a <code>hdrs</code> attribute. All headers that belong to the binary or test
  directly should be listed in the <code>srcs</code>.
</p>

<p>
  To illustrate these rules, look at the following example.
</p>

<pre class="code">
cc_binary(
    name = "foo",
    srcs = [
        "foo.cc",
        "foo.h",
    ],
    deps = [":bar"],
)

cc_library(
    name = "bar",
    srcs = [
        "bar.cc",
        "bar-impl.h",
    ],
    hdrs = ["bar.h"],
    deps = [":baz"],
)

cc_library(
    name = "baz",
    srcs = [
        "baz.cc",
        "baz-impl.h",
    ],
    hdrs = ["baz.h"],
)
</pre>

<p>
  The allowed direct inclusions in this example are listed in the table below. For example
  <code>foo.cc</code> is allowed to directly include <code>foo.h</code> and <code>bar.h</code>, but
  not <code>baz.h</code>.
</p>

<table class="table table-striped table-bordered table-condensed">
  <thead>
    <tr><th>Including file</th><th>Allowed inclusions</th></tr>
  </thead>
  <tbody>
    <tr><td>foo.h</td><td>bar.h</td></tr>
    <tr><td>foo.cc</td><td>foo.h bar.h</td></tr>
    <tr><td>bar.h</td><td>bar-impl.h baz.h</td></tr>
    <tr><td>bar-impl.h</td><td>bar.h baz.h</td></tr>
    <tr><td>bar.cc</td><td>bar.h bar-impl.h baz.h</td></tr>
    <tr><td>baz.h</td><td>baz-impl.h</td></tr>
    <tr><td>baz-impl.h</td><td>baz.h</td></tr>
    <tr><td>baz.cc</td><td>baz.h baz-impl.h</td></tr>
  </tbody>
</table>

<p>
  The inclusion checking rules only apply to <em>direct</em>
  inclusions. In the example above <code>foo.cc</code> is allowed to
  include <code>bar.h</code>, which may include <code>baz.h</code>, which in
  turn is allowed to include <code>baz-impl.h</code>. Technically, the
  compilation of a <code>.cc</code> file may transitively include any header
  file in the <code>hdrs</code> or <code>srcs</code> in
  any <code>cc_library</code> in the transitive <code>deps</code> closure. In
  this case the compiler may read <code>baz.h</code> and <code>baz-impl.h</code>
  when compiling <code>foo.cc</code>, but <code>foo.cc</code> must not
  contain <code>#include "baz.h"</code>. For that to be
  allowed, <code>baz</code> must be added to the <code>deps</code>
  of <code>foo</code>.
</p>

<p>
  Unfortunately Bazel currently cannot distinguish between direct and transitive
  inclusions, so it cannot detect error cases where a file illegally includes a
  header directly that is only allowed to be included transitively. For example,
  Bazel would not complain if in the example above <code>foo.cc</code> directly
  includes <code>baz.h</code>. This would be illegal, because <code>foo</code>
  does not directly depend on <code>baz</code>. Currently, no error is produced
  in that case, but such error checking may be added in the future.
</p>

<!-- #END_BLAZE_RULE -->*/


/*<!-- #BLAZE_RULE (NAME = cc_test, TYPE = TEST, FAMILY = C / C++) -->

<!-- #END_BLAZE_RULE -->*/
