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
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainTransitionMode;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses.CcIncludeScanningRule;
import com.google.devtools.build.lib.rules.cpp.GraphNodeAspect;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Rule class definitions for C++ rules.
 */
public class BazelCppRuleClasses {
  static final SafeImplicitOutputsFunction CC_LIBRARY_DYNAMIC_LIB =
      fromTemplates("%{dirname}lib%{basename}.so");

  static final SafeImplicitOutputsFunction CC_BINARY_IMPLICIT_OUTPUTS =
      fromFunctions(CppRuleClasses.CC_BINARY_STRIPPED, CppRuleClasses.CC_BINARY_DEBUG_PACKAGE);

  static final FileTypeSet ALLOWED_SRC_FILES =
      FileTypeSet.of(
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

  static final String[] DEPS_ALLOWED_RULES =
      new String[] {
        "cc_library", "objc_library", "cc_proto_library", "cc_import",
      };

  /** Common attributes for all rules that need a C++ toolchain. */
  public static final class CcToolchainRequiringRule implements RuleDefinition {
    @Override
    @SuppressWarnings("unchecked")
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(
              attr(CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME, LABEL)
                  .mandatoryProviders(CcToolchainProvider.PROVIDER.id())
                  .value(CppRuleClasses.ccToolchainAttribute(env)))
          .add(
              attr(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL)
                  .value(CppRuleClasses.ccToolchainTypeAttribute(env)))
          .setPreferredDependencyPredicate(Predicates.<String>or(CPP_SOURCE, C_SOURCE, CPP_HEADER))
          .requiresConfigurationFragments(PlatformConfiguration.class)
          .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(env))
          .useToolchainTransition(ToolchainTransitionMode.ENABLED)
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_toolchain_requiring_rule")
          .ancestors(CcIncludeScanningRule.class)
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }

  /**
   * Common attributes for C++ rules.
   */
  public static final class CcBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /*<!-- #BLAZE_RULE($cc_base_rule).ATTRIBUTE(copts) -->
          Add these options to the C++ compilation command.
          Subject to <a href="${link make-variables}">"Make variable"</a> substitution and
          <a href="${link common-definitions#sh-tokenization}">Bourne shell tokenization</a>.
          <p>
            Each string in this attribute is added in the given order to <code>COPTS</code> before
            compiling the binary target. The flags take effect only for compiling this target, not
            its dependencies, so be careful about header files included elsewhere. All paths should
            be relative to the workspace, not to the current package.
          </p>
          <p>
            If the package declares the <a href="${link package.features}">feature</a>
            <code>no_copts_tokenization</code>, Bourne shell tokenization applies only to strings
            that consist of a single "Make" variable.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("copts", STRING_LIST))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_base_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(CcToolchainRequiringRule.class)
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  public static final class CcDeclRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /*<!-- #BLAZE_RULE($cc_decl_rule).ATTRIBUTE(defines) -->
          List of defines to add to the compile line.
          Subject to <a href="${link make-variables}">"Make" variable</a> substitution and
          <a href="${link common-definitions#sh-tokenization}">Bourne shell tokenization</a>.
          Each string, which must consist of a single Bourne shell token,
          is prepended with <code>-D</code> and added to the compile command line to this target,
          as well as to every rule that depends on it. Be very careful, since this may have
          far-reaching effects. When in doubt, add define values to
          <a href="#cc_binary.local_defines"><code>local_defines</code></a> instead.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("defines", STRING_LIST))
          /*<!-- #BLAZE_RULE($cc_decl_rule).ATTRIBUTE(local_defines) -->
          List of defines to add to the compile line.
          Subject to <a href="${link make-variables}">"Make" variable</a> substitution and
          <a href="${link common-definitions#sh-tokenization}">Bourne shell tokenization</a>.
          Each string, which must consist of a single Bourne shell token,
          is prepended with <code>-D</code> and added to the compile command line for this target,
          but not to its dependents.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("local_defines", STRING_LIST))
          /*<!-- #BLAZE_RULE($cc_decl_rule).ATTRIBUTE(includes) -->
          List of include dirs to be added to the compile line.
          <p>
          Subject to <a href="${link make-variables}">"Make variable"</a> substitution.
          Each string is prepended with <code>-isystem</code> and added to <code>COPTS</code>.
          Unlike <a href="#cc_binary.copts">COPTS</a>, these flags are added for this rule
          and every rule that depends on it. (Note: not the rules it depends upon!) Be
          very careful, since this may have far-reaching effects. When in doubt, add
          "-I" flags to <a href="#cc_binary.copts">COPTS</a> instead.
          </p>
          <p>
          Headers must be added to srcs or hdrs, otherwise they will not be available to dependent
          rules when compilation is sandboxed (the default).
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("includes", STRING_LIST))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_decl_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
          .build();
    }
  }

  /**
   * Helper rule class.
   */
  public static final class CcRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
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
              <code>.cxx</code>, <code>.c++</code>, <code>.C</code></li>
            <li>C and C++ header files: <code>.h</code>, <code>.hh</code>, <code>.hpp</code>,
              <code>.hxx</code>, <code>.inc</code>, <code>.inl</code>, <code>.H</code></li>
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
          <p>These can be <code>cc_library</code> or <code>objc_library</code>
          targets.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .override(
              attr("deps", LABEL_LIST)
                  .allowedRuleClasses(DEPS_ALLOWED_RULES)
                  .allowedFileTypes(CppFileTypes.LINKER_SCRIPT)
                  .skipAnalysisTimeFileTypeCheck()
                  .mandatoryProviders(StarlarkProviderIdentifier.forKey(CcInfo.PROVIDER.getKey())))
          /*<!-- #BLAZE_RULE($cc_rule).ATTRIBUTE(win_def_file) -->
          The Windows DEF file to be passed to linker.
          <p>This attribute should only be used when Windows is the target platform.
          It can be used to <a href="https://msdn.microsoft.com/en-us/library/d91k01sh.aspx">
          export symbols</a> during linking a shared library.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("win_def_file", LABEL).allowedFileTypes(CppFileTypes.WINDOWS_DEF_FILE))
          .add(
              attr("reexport_deps", LABEL_LIST)
                  .allowedRuleClasses(DEPS_ALLOWED_RULES)
                  .allowedFileTypes())
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
            assumed to be the label of a target in <code>deps</code>. The
            list of files generated by that target is appended to the linker
            options. An error is reported if the label is invalid, or is
            not declared in <code>deps</code>.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("linkopts", STRING_LIST))
          /*<!-- #BLAZE_RULE($cc_rule).ATTRIBUTE(nocopts) -->
          Remove matching options from the C++ compilation command.
          Subject to <a href="${link make-variables}">"Make" variable</a> substitution.
          The value of this attribute is interpreted as a regular expression.
          Any preexisting <code>COPTS</code> that match this regular expression
          (including values explicitly specified in the rule's <a
          href="#cc_binary.copts">copts</a> attribute) will be removed from
          <code>COPTS</code> for purposes of compiling this rule.
          This attribute should rarely be needed.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("nocopts", STRING))
          /*<!-- #BLAZE_RULE($cc_rule).ATTRIBUTE(linkstatic) -->
           For <a href="${link cc_binary}"><code>cc_binary</code></a> and
           <a href="${link cc_test}"><code>cc_test</code></a>: link the binary in static
           mode. For <code>cc_library.linkstatic</code>: see below.
           <p>
             By default this option is on for <code>cc_binary</code> and off for the rest.
           </p>
           <p>
             If enabled and this is a binary or test, this option tells the build tool to link in
             <code>.a</code>'s instead of <code>.so</code>'s for user libraries whenever possible.
             Some system libraries may still be linked dynamically, as are libraries for which
             there is no static library. So the resulting executable will still be dynamically
             linked, hence only <i>mostly</i> static.
           </p>
           <p>There are really three different ways to link an executable:</p>
           <ul>
           <li> STATIC with fully_static_link feature, in which everything is linked statically;
             e.g. "<code>gcc -static foo.o libbar.a libbaz.a -lm</code>".<br/>
             This mode is enabled by specifying <code>fully_static_link</code> in the
             <a href="${link common-definitions#features}"><code>features</code></a> attribute.</li>
           <li> STATIC, in which all user libraries are linked statically (if a static
             version is available), but where system libraries (excluding C/C++ runtime libraries)
             are linked dynamically, e.g. "<code>gcc foo.o libfoo.a libbaz.a -lm</code>".<br/>
             This mode is enabled by specifying <code>linkstatic=True</code>.</li>
           <li> DYNAMIC, in which all libraries are linked dynamically (if a dynamic version is
             available), e.g. "<code>gcc foo.o libfoo.so libbaz.so -lm</code>".<br/>
             This mode is enabled by specifying <code>linkstatic=False</code>.</li>
           </ul>
           <p>
           The <code>linkstatic</code> attribute has a different meaning if used on a
           <a href="${link cc_library}"><code>cc_library()</code></a> rule.
           For a C++ library, <code>linkstatic=True</code> indicates that only
           static linking is allowed, so no <code>.so</code> will be produced. linkstatic=False does
           not prevent static libraries from being created. The attribute is meant to control the
           creation of dynamic libraries.
           </p>
           <p>
           If <code>linkstatic=False</code>, then the build tool will create symlinks to
           depended-upon shared libraries in the <code>*.runfiles</code> area.
           </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("linkstatic", BOOLEAN).value(true))
          .add(
              attr("$def_parser", LABEL)
                  .cfg(HostTransition.createFactory())
                  .singleArtifact()
                  .value(
                      new Attribute.ComputedDefault() {
                        @Override
                        public Object getDefault(AttributeMap rule) {
                          // Every cc_rule depends implicitly on the def_parser tool.
                          // The only exceptions are the rules for building def_parser itself.
                          // To avoid cycles in the dependency graph, return null for rules under
                          // @bazel_tools//third_party/def_parser and @bazel_tools//tools/cpp
                          String label = rule.getLabel().toString();
                          String toolsRepository = env.getToolsRepository();
                          return label.startsWith(toolsRepository + "//third_party/def_parser")
                                  // @bazel_tools//tools/cpp:malloc and @bazel_tools//tools/cpp:stl
                                  // are implicit dependencies of all cc rules,
                                  // thus a dependency of the def_parser.
                                  || label.startsWith(toolsRepository + "//tools/cpp")
                              ? null
                              : env.getToolsLabel("//tools/def_parser:def_parser");
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
  public static final class CcLibraryBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
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
          .add(
              attr("hdrs", LABEL_LIST)
                  .orderIndependent()
                  .direct_compile_time_input()
                  .allowedFileTypes(FileTypeSet.ANY_FILE))
          /* <!-- #BLAZE_RULE($cc_library).ATTRIBUTE(strip_include_prefix) -->
          The prefix to strip from the paths of the headers of this rule.

          <p>When set, the headers in the <code>hdrs</code> attribute of this rule are accessible
          at their path with this prefix cut off.

          <p>If it's a relative path, it's taken as a package-relative one. If it's an absolute one,
          it's understood as a repository-relative path.

          <p>The prefix in the <code>include_prefix</code> attribute is added after this prefix is
          stripped.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("strip_include_prefix", STRING))
          /* <!-- #BLAZE_RULE($cc_library).ATTRIBUTE(include_prefix) -->
          The prefix to add to the paths of the headers of this rule.

          <p>When set, the headers in the <code>hdrs</code> attribute of this rule are accessible
          at is the value of this attribute prepended to their repository-relative path.

          <p>The prefix in the <code>strip_include_prefix</code> attribute is removed before this
          prefix is added.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("include_prefix", STRING))
          /* <!-- #BLAZE_RULE($cc_library).ATTRIBUTE(textual_hdrs) -->
           The list of header files published by
           this library to be textually included by sources in dependent rules.
          <p>This is the location for declaring header files that cannot be compiled on their own;
             that is, they always need to be textually included by other source files to build valid
             code.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(
              attr("textual_hdrs", LABEL_LIST)
                  .orderIndependent()
                  .direct_compile_time_input()
                  .legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($cc_library).ATTRIBUTE(linkstamp) -->
          Simultaneously compiles and links the specified C++ source file into the final
          binary. This trickery is required to introduce timestamp
          information into binaries; if we compiled the source file to an
          object file in the usual way, the timestamp would be incorrect.
          A linkstamp compilation may not include any particular set of
          compiler flags and so should not depend on any particular
          header, compiler option, or other build variable.
          <em class='harmful'>This option should only be needed in the
          <code>base</code> package.</em>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
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

  /** Helper rule class. */
  public static final class CcBinaryBaseRule implements RuleDefinition {
    private final GraphNodeAspect graphNodeAspect;

    public CcBinaryBaseRule(GraphNodeAspect graphNodeAspect) {
      this.graphNodeAspect = graphNodeAspect;
    }

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /*<!-- #BLAZE_RULE($cc_binary_base).ATTRIBUTE(additional_linker_inputs) -->
           Pass these files to the C++ linker command.
           <p>
           For example, compiled Windows .res files can be provided here to be embedded in
           the binary target.
           </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(
              attr("additional_linker_inputs", LABEL_LIST)
                  .orderIndependent()
                  .direct_compile_time_input()
                  .allowedFileTypes(FileTypeSet.ANY_FILE))
          .override(
              attr("deps", LABEL_LIST)
                  .allowedRuleClasses(DEPS_ALLOWED_RULES)
                  .allowedFileTypes(CppFileTypes.LINKER_SCRIPT)
                  .skipAnalysisTimeFileTypeCheck()
                  .mandatoryProviders(StarlarkProviderIdentifier.forKey(CcInfo.PROVIDER.getKey()))
                  .aspect(graphNodeAspect, GraphNodeAspect.ASPECT_PARAMETERS))
          .add(
              attr("dynamic_deps", LABEL_LIST)
                  .allowedFileTypes(FileTypeSet.NO_FILE)
                  .mandatoryProvidersList(
                      ImmutableList.of(
                          ImmutableList.of(
                              StarlarkProviderIdentifier.forKey(
                                  BazelCppSemantics.CC_SHARED_INFO_PROVIDER)),
                          ImmutableList.of(
                              StarlarkProviderIdentifier.forKey(
                                  BazelCppSemantics.CC_SHARED_INFO_PROVIDER_RULES_CC)))))
          /*<!-- #BLAZE_RULE($cc_binary_base).ATTRIBUTE(malloc) -->
          Override the default dependency on malloc.
          <p>
            By default, C++ binaries are linked against <code>//tools/cpp:malloc</code>,
            which is an empty library so the binary ends up using libc malloc.
            This label must refer to a <code>cc_library</code>. If compilation is for a non-C++
            rule, this option has no effect. The value of this attribute is ignored if
            <code>linkshared=True</code> is specified.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(
              attr("malloc", LABEL)
                  .value(env.getToolsLabel("//tools/cpp:malloc"))
                  .allowedFileTypes()
                  .allowedRuleClasses("cc_library")
                  .aspect(graphNodeAspect, GraphNodeAspect.ASPECT_PARAMETERS))
          .add(attr(":default_malloc", LABEL).value(CppRuleClasses.DEFAULT_MALLOC))
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
              by the <a href="../user-manual.html#flag--stamp">--[no]stamp</a> flag.</li>
          </ul>
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .add(attr("stamp", TRISTATE).value(TriState.AUTO))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$cc_binary_base")
          .type(RuleClassType.ABSTRACT)
          .ancestors(CcRule.class)
          .build();
    }
  }
}
