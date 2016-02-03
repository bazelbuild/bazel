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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.PredicateWithMessage;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleClass.PackageNameConstraint;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.Set;

/**
 * Rule class definitions for Java rules.
 */
public class BazelJavaRuleClasses {

  public static final PredicateWithMessage<Rule> JAVA_PACKAGE_NAMES = new PackageNameConstraint(
      PackageNameConstraint.ANY_SEGMENT, "java", "javatests");

  protected static final String JUNIT_TESTRUNNER = "//tools/jdk:TestRunner_deploy.jar";

  public static final ImplicitOutputsFunction JAVA_BINARY_IMPLICIT_OUTPUTS =
      fromFunctions(
          JavaSemantics.JAVA_BINARY_CLASS_JAR,
          JavaSemantics.JAVA_BINARY_SOURCE_JAR,
          JavaSemantics.JAVA_BINARY_DEPLOY_JAR,
          JavaSemantics.JAVA_BINARY_DEPLOY_SOURCE_JAR);

  static final ImplicitOutputsFunction JAVA_LIBRARY_IMPLICIT_OUTPUTS =
      fromFunctions(
          JavaSemantics.JAVA_LIBRARY_CLASS_JAR,
          JavaSemantics.JAVA_LIBRARY_SOURCE_JAR);

  /**
   * Common attributes for rules that depend on ijar.
   */
  public static final class IjarBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("$ijar", LABEL).cfg(HOST).exec().value(env.getLabel("//tools/defaults:ijar")))
          .setPreferredDependencyPredicate(JavaSemantics.JAVA_SOURCE)
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$ijar_base_rule")
          .type(RuleClassType.ABSTRACT)
          .build();
    }
  }


  /**
   * Common attributes for Java rules.
   */
  public static final class JavaBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr(":jvm", LABEL).cfg(HOST).value(JavaSemantics.JVM))
          .add(attr(":host_jdk", LABEL).cfg(HOST).value(JavaSemantics.HOST_JDK))
          .add(attr(":java_toolchain", LABEL).value(JavaSemantics.JAVA_TOOLCHAIN))
          .add(attr("$javac_extdir", LABEL).cfg(HOST)
              .value(env.getLabel(JavaSemantics.JAVAC_EXTDIR_LABEL)))
          .add(attr("$java_langtools", LABEL).cfg(HOST)
              .value(env.getLabel("//tools/defaults:java_langtools")))
          .add(attr("$javac_bootclasspath", LABEL).cfg(HOST)
              .value(env.getLabel(JavaSemantics.JAVAC_BOOTCLASSPATH_LABEL)))
          .add(attr("$javabuilder", LABEL).cfg(HOST)
              .value(env.getLabel(JavaSemantics.JAVABUILDER_LABEL)))
          .add(attr("$singlejar", LABEL).cfg(HOST)
              .value(env.getLabel(JavaSemantics.SINGLEJAR_LABEL)))
          .add(attr("$genclass", LABEL).cfg(HOST)
              .value(env.getLabel(JavaSemantics.GENCLASS_LABEL)))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$java_base_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(IjarBaseRule.class)
          .build();
    }
  }

  static final Set<String> ALLOWED_RULES_IN_DEPS = ImmutableSet.of(
      "cc_binary",  // NB: linkshared=1
      "cc_library",
      "genrule",
      "genproto",  // TODO(bazel-team): we should filter using providers instead (skylark rule).
      "java_import",
      "java_library",
      // There is no Java protoc for Bazel--yet. This is here for the benefit of J2 protos.
      "proto_library",
      "sh_binary",
      "sh_library");

  /**
   * Common attributes for Java rules.
   */
  public static final class JavaRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($java_rule).ATTRIBUTE(deps) -->
          The list of other libraries to be linked in to the target.
          See general comments about <code>deps</code> at
          <a href="common-definitions.html#common-attributes">Attributes common to all build rules
          </a>.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .override(builder.copy("deps")
              .allowedFileTypes(JavaSemantics.JAR)
              .allowedRuleClasses(ALLOWED_RULES_IN_DEPS)
              .skipAnalysisTimeFileTypeCheck())
          /* <!-- #BLAZE_RULE($java_rule).ATTRIBUTE(runtime_deps) -->
          Libraries to make available to the final binary or test at runtime only.
          Like ordinary <code>deps</code>, these will appear on the runtime classpath, but unlike
          them, not on the compile-time classpath. Dependencies needed only at runtime should be
          listed here. Dependency-analysis tools should ignore targets that appear in both
          <code>runtime_deps</code> and <code>deps</code>.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("runtime_deps", LABEL_LIST)
              .allowedFileTypes(JavaSemantics.JAR)
              .allowedRuleClasses(ALLOWED_RULES_IN_DEPS)
              .skipAnalysisTimeFileTypeCheck())

          /* <!-- #BLAZE_RULE($java_rule).ATTRIBUTE(srcs) -->
          The list of source files that are processed to create the target.
          This attribute is almost always required; see exceptions below.
          <p>
            Source files of type <code>.java</code> are compiled. In case of generated
            <code>.java</code> files it is generally advisable to put the generating rule's name
            here instead of the name of the file itself. This not only improves readability but
            makes the rule more resilient to future changes: if the generating rule generates
            different files in the future, you only need to fix one place: the <code>outs</code> of
            the generating rule. You should not list the generating rule in <code>deps</code>
            because it is a no-op.
          </p>
          <p>
            Source files of type <code>.srcjar</code> are unpacked and compiled. (This is useful if
            you need to generate a set of <code>.java</code> files with a genrule.)
          </p>
          <p>
            Rules: if the rule (typically <code>genrule</code> or <code>filegroup</code>) generates
            any of the files listed above, they will be used the same way as described for source
            files.
          </p>

          <p>
            This argument is almost always required, except if a
            <a href="#java_binary.main_class"><code>main_class</code></a> attribute specifies a
            class on the runtime classpath or you specify the <code>runtime_deps</code> argument.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("srcs", LABEL_LIST)
              .orderIndependent()
              .direct_compile_time_input()
              .allowedFileTypes(JavaSemantics.JAVA_SOURCE,
                  JavaSemantics.SOURCE_JAR, JavaSemantics.PROPERTIES))
          /* <!-- #BLAZE_RULE($java_rule).ATTRIBUTE(resources) -->
          A list of data files to include in a Java jar.
          <p>
            If resources are specified, they will be bundled in the jar along with the usual
            <code>.class</code> files produced by compilation. The location of the resources inside
            of the jar file is determined by the project structure. Bazel first looks for Maven's
            <a href="https://maven.apache.org/guides/introduction/introduction-to-the-standard-directory-layout.html">standard directory layout</a>,
            (a "src" directory followed by a "resources" directory grandchild). If that is not
            found, Bazel then looks for the topmost directory named "java" or "javatests" (so, for
            example, if a resource is at &lt;workspace root&gt;/x/java/y/java/z, Bazel will use the
             path y/java/z. This heuristic cannot be overridden.
          </p>

          <p>
            Resources may be source files or generated files.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("resources", LABEL_LIST).orderIndependent()
              .allowedFileTypes(FileTypeSet.ANY_FILE))
          /* <!-- #BLAZE_RULE($java_rule).ATTRIBUTE(resource_strip_prefix) -->
          The path prefix to strip from Java resources.
          <p>
            If specified, this path prefix is stripped from every file in the <code>resources</code>
            attribute. It is an error for a resource file not to be under this directory. If not
            specified (the default), the path of resource file is determined according to the same
            logic as the Java package of source files. For example, a source file at
            <code>stuff/java/foo/bar/a.txt</code> will be located at <code>foo/bar/a.txt</code>.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("resource_strip_prefix", STRING))
          /* <!-- #BLAZE_RULE($java_rule).ATTRIBUTE(plugins) -->
          Java compiler plugins to run at compile-time.
          Every <code>java_plugin</code> specified in this attribute will be run whenever this rule
          is built. A library may also inherit plugins from dependencies that use
          <code><a href="#java_library.exported_plugins">exported_plugins</a></code>. Resources
          generated by the plugin will be included in the resulting jar of this rule.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("plugins", LABEL_LIST).cfg(HOST).allowedRuleClasses("java_plugin")
              .legacyAllowAnyFileType())
          .add(attr(":java_plugins", LABEL_LIST)
              .cfg(HOST)
              .allowedRuleClasses("java_plugin")
              .silentRuleClassFilter()
              .value(JavaSemantics.JAVA_PLUGINS))
          /* <!-- #BLAZE_RULE($java_rule).ATTRIBUTE(javacopts) -->
          Extra compiler options for this library.
          Subject to <a href="make-variables.html">"Make variable"</a> substitution and
          <a href="common-definitions.html#sh-tokenization">Bourne shell tokenization</a>.
          <p>These compiler options are passed to javac after the global compiler options.</p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("javacopts", STRING_LIST))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$java_rule")
          .type(RuleClassType.ABSTRACT)
          .ancestors(BaseRuleClasses.RuleBase.class, JavaBaseRule.class)
          .build();
    }
  }

  /**
   * Base class for rule definitions producing Java binaries.
   */
  public static final class BaseJavaBinaryRule implements RuleDefinition {

    @Override
    public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
      return builder
          /* <!-- #BLAZE_RULE($base_java_binary).ATTRIBUTE(classpath_resources) -->
          <em class="harmful">DO NOT USE THIS OPTION UNLESS THERE IS NO OTHER WAY)</em>
          <p>
            A list of resources that must be located at the root of the java tree. This attribute's
            only purpose is to support third-party libraries that require that their resources be
            found on the classpath as exactly <code>"myconfig.xml"</code>. It is only allowed on
            binaries and not libraries, due to the danger of namespace conflicts.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("classpath_resources", LABEL_LIST).legacyAllowAnyFileType())
          /* <!-- #BLAZE_RULE($base_java_binary).ATTRIBUTE(jvm_flags) -->
          A list of flags to embed in the wrapper script generated for running this binary.
          Subject to <a href="#location">$(location)</a> and
          <a href="make-variables.html">"Make variable"</a> substitution, and
          <a href="common-definitions.html#sh-tokenization">Bourne shell tokenization</a>.
          <p>
            The wrapper script for a Java binary includes a <code>CLASSPATH</code> definition (to
            find all the dependent jars) and invokes the right Java interpreter. The command line
            generated by the wrapper script includes the name of the main class followed by a
            <code>"$@"</code> so you can pass along other arguments after the classname.  However,
            arguments intended for parsing by the JVM must be specified <i>before</i> the classname
            on the command line. The contents of <code>jvm_flags</code> are added to the wrapper
            script before the classname is listed.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("jvm_flags", STRING_LIST))
          /* <!-- #BLAZE_RULE($base_java_binary).ATTRIBUTE(use_testrunner) -->
          Use the
          <code>com.google.testing.junit.runner.GoogleTestRunner</code> class as the
          main entry point for a Java program.

          You can use this to override the default
          behavior, which is to use <code>BazelTestRunner</code> for
          <code>java_test</code> rules,
          and not use it for <code>java_binary</code> rules.  It is unlikely
          you will want to do this.  One use is for <code>AllTest</code>
          rules that are invoked by another rule (to set up a database
          before running the tests, for example).  The <code>AllTest</code>
          rule must be declared as a <code>java_binary</code>, but should
          still use the test runner as its main entry point.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("use_testrunner", BOOLEAN).value(false))
          /* <!-- #BLAZE_RULE($base_java_binary).ATTRIBUTE(main_class) -->
          Name of class with <code>main()</code> method to use as entry point.
          If a rule uses this option, it does not need a <code>srcs=[...]</code> list.
          Thus, with this attribute one can make an executable from a Java library that already
          contains one or more <code>main()</code> methods.
          <p>
            The value of this attribute is a class name, not a source file. The class must be
            available at runtime: it may be compiled by this rule (from <code>srcs</code>) or
            provided by direct or transitive dependencies (through <code>runtime_deps</code> or
            <code>deps</code>). If the class is unavailable, the binary will fail at runtime; there
            is no build-time check.
          </p>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("main_class", STRING))
          /* <!-- #BLAZE_RULE($base_java_binary).ATTRIBUTE(create_executable) -->
          Whether to build the executable wrapper script or not.
          If this option is present, no executable wrapper script is built around the
          <code>.jar</code> file. Incompatible with <code>main_class</code> attribute.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("create_executable", BOOLEAN)
              .nonconfigurable("internal")
              .value(true))
          .add(attr("$testsupport", LABEL).value(
              new Attribute.ComputedDefault("use_testrunner") {
                @Override
                public Object getDefault(AttributeMap rule) {
                  return rule.get("use_testrunner", Type.BOOLEAN)
                    ? env.getLabel(env.getToolsRepository() + JUNIT_TESTRUNNER)
                    : null;
                }
              }))
          /* <!-- #BLAZE_RULE($base_java_binary).ATTRIBUTE(deploy_manifest_lines) -->
          A list of lines to add to the <code>META-INF/manifest.mf</code> file generated for the
          <code>*_deploy.jar</code> target. The contents of this attribute are <em>not</em> subject
          to <a href="make-variables.html">"Make variable"</a> substitution.
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          .add(attr("deploy_manifest_lines", STRING_LIST))
          /* <!-- #BLAZE_RULE($base_java_binary).ATTRIBUTE(stamp) -->
          Enable link stamping.
          Whether to encode build information into the binary. Possible values:
          <ul>
            <li><code>stamp = 1</code>: Stamp the build information into the binary. Stamped
              binaries are only rebuilt when their dependencies change. Use this if there are tests
              that depend on the build information.</li>
            <li><code>stamp = 0</code>: Always replace build information by constant values. This
              gives good build result caching.</li>
            <li><code>stamp = -1</code>: Embedding of build information is controlled by the
              <a href="../blaze-user-manual.html#flag--stamp">--[no]stamp</a> flag.</li>
          </ul>
          <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
          // TODO(bazel-team): describe how to access this data at runtime
          .add(attr("stamp", TRISTATE).value(TriState.AUTO))
          .add(attr(":java_launcher", LABEL).value(JavaSemantics.JAVA_LAUNCHER))  // blaze flag
          .build();
    }
    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("$base_java_binary")
          .type(RuleClassType.ABSTRACT)
          .ancestors(JavaRule.class,
              // java_binary and java_test require the crosstool C++ runtime
              // libraries (libstdc++.so, libgcc_s.so).
              // TODO(bazel-team): Add tests for Java+dynamic runtime.
              BazelCppRuleClasses.CcLinkingRule.class)
          .build();
    }
  }
}
