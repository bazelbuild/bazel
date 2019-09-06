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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigAwareRuleClassBuilder;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses.JavaRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.ProguardLibraryRule;

/**
 * Common attributes for Java rules.
 */
public final class BazelJavaLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {

    return ConfigAwareRuleClassBuilder.of(builder)
        // For getting the host Java executable.
        .requiresHostConfigurationFragments(JavaConfiguration.class)
        .originalBuilder()
        .requiresConfigurationFragments(JavaConfiguration.class, CppConfiguration.class)
        /* <!-- #BLAZE_RULE(java_library).IMPLICIT_OUTPUTS -->
        <ul>
          <li><code>lib<var>name</var>.jar</code>: A Java archive containing the class files.</li>
          <li><code>lib<var>name</var>-src.jar</code>: An archive containing the sources ("source
            jar").</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS --> */
        .setImplicitOutputsFunction(BazelJavaRuleClasses.JAVA_LIBRARY_IMPLICIT_OUTPUTS)

        /* <!-- #BLAZE_RULE(java_library).ATTRIBUTE(data) -->
        The list of files needed by this library at runtime.
        See general comments about <code>data</code> at
        <a href="${link common-definitions#common-attributes}">Attributes common to all build rules
        </a>.
        <p>
          When building a <code>java_library</code>, Bazel doesn't put these files anywhere; if the
          <code>data</code> files are generated files then Bazel generates them. When building a
          test that depends on this <code>java_library</code> Bazel copies or links the
          <code>data</code> files into the runfiles area.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */

        /* <!-- #BLAZE_RULE(java_library).ATTRIBUTE(deps) -->
        The list of libraries to link into this library.
        See general comments about <code>deps</code> at
        <a href="${link common-definitions#common-attributes}">Attributes common to all build rules
        </a>.
        <p>
          The jars built by <code>java_library</code> rules listed in <code>deps</code> will be on
          the compile-time classpath of this rule. Furthermore the transitive closure of their
          <code>deps</code>, <code>runtime_deps</code> and <code>exports</code> will be on the
          runtime classpath.
        </p>
        <p>
          By contrast, targets in the <code>data</code> attribute are included in the runfiles but
          on neither the compile-time nor runtime classpath.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */

        /* <!-- #BLAZE_RULE(java_library).ATTRIBUTE(exports) -->
        Exported libraries.
        <p>
          Listing rules here will make them available to parent rules, as if the parents explicitly
          depended on these rules. This is not true for regular (non-exported) <code>deps</code>.
        </p>
        <p>
          Summary: a rule <i>X</i> can access the code in <i>Y</i> if there exists a dependency
          path between them that begins with a <code>deps</code> edge followed by zero or more
          <code>exports</code> edges. Let's see some examples to illustrate this.
        </p>
        <p>
          Assume <i>A</i> depends on <i>B</i> and <i>B</i> depends on <i>C</i>. In this case
          C is a <em>transitive</em> dependency of A, so changing C's sources and rebuilding A will
          correctly rebuild everything. However A will not be able to use classes in C. To allow
          that, either A has to declare C in its <code>deps</code>, or B can make it easier for A
          (and anything that may depend on A) by declaring C in its (B's) <code>exports</code>
          attribute.
        </p>
        <p>
          The closure of exported libraries is available to all direct parent rules. Take a slightly
          different example: A depends on B, B depends on C and D, and also exports C but not D.
          Now A has access to C but not to D. Now, if C and D exported some libraries, C' and D'
          respectively, A could only access C' but not D'.
        </p>
        <p>
          Important: an exported rule is not a regular dependency. Sticking to the previous example,
          if B exports C and wants to also use C, it has to also list it in its own
          <code>deps</code>.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("exports", LABEL_LIST)
                .allowedRuleClasses(BazelJavaRuleClasses.ALLOWED_RULES_IN_DEPS)
                .allowedFileTypes(/*May not have files in exports!*/ )
                .mandatoryProvidersList(BazelJavaRuleClasses.MANDATORY_JAVA_PROVIDER_ONLY))
        /* <!-- #BLAZE_RULE(java_library).ATTRIBUTE(neverlink) -->
        Whether this library should only be used for compilation and not at runtime.
        Useful if the library will be provided by the runtime environment during execution. Examples
        of such libraries are the IDE APIs for IDE plug-ins or <code>tools.jar</code> for anything
        running on a standard JDK.
        <p>
          Note that <code>neverlink = 1</code> does not prevent the compiler from inlining material
          from this library into compilation targets that depend on it, as permitted by the Java
          Language Specification (e.g., <code>static final</code> constants of <code>String</code>
          or of primitive types). The preferred use case is therefore when the runtime library is
          identical to the compilation library.
        </p>
        <p>
          If the runtime library differs from the compilation library then you must ensure that it
          differs only in places that the JLS forbids compilers to inline (and that must hold for
          all future versions of the JLS).
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("neverlink", BOOLEAN).value(false))
        .override(attr("javacopts", STRING_LIST))
        /* <!-- #BLAZE_RULE(java_library).ATTRIBUTE(exported_plugins) -->
        The list of <code><a href="#${link java_plugin}">java_plugin</a></code>s (e.g. annotation
        processors) to export to libraries that directly depend on this library.
        <p>
          The specified list of <code>java_plugin</code>s will be applied to any library which
          directly depends on this library, just as if that library had explicitly declared these
          labels in <code><a href="${link java_library.plugins}">plugins</a></code>.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("exported_plugins", LABEL_LIST)
                .cfg(HostTransition.createFactory())
                .allowedRuleClasses("java_plugin")
                .allowedFileTypes())
        .advertiseSkylarkProvider(SkylarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey()))
        .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(env))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_library")
        .ancestors(JavaRule.class, ProguardLibraryRule.class)
        .factoryClass(BazelJavaLibrary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = java_library, TYPE = LIBRARY, FAMILY = Java) -->

<p>This rule compiles and links sources into a <code>.jar</code> file.</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
