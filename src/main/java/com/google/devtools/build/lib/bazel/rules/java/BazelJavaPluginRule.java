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
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for the java_plugin rule.
 */
public final class BazelJavaPluginRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(java_plugin).IMPLICIT_OUTPUTS -->
        <ul>
          <li><code><var>libname</var>.jar</code>: A Java archive.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS --> */
        .setImplicitOutputsFunction(BazelJavaRuleClasses.JAVA_LIBRARY_IMPLICIT_OUTPUTS)
        .override(builder.copy("deps").validityPredicate(Attribute.ANY_EDGE))
        .override(builder.copy("srcs").validityPredicate(Attribute.ANY_EDGE))
        /* <!-- #BLAZE_RULE(java_plugin).ATTRIBUTE(processor_class) -->
        ${SYNOPSIS}
        The processor class is the fully qualified type of the class that the Java compiler should
        use as entry point to the annotation processor. If not specified, this rule will not
        contribute an annotation processor to the Java compiler's annotation processing, but its
        runtime classpath will still be included on the compiler's annotation processor path. (This
        is primarily intended for use by <a href="http://errorprone.info/">Error Prone plugins</a>,
        which are loaded from the annotation processor path using <code>META-INF/services</code>
        files.)
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("processor_class", STRING))
        .removeAttribute("runtime_deps")
        .removeAttribute("exports")
        .removeAttribute("exported_plugins")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_plugin")
        .ancestors(BazelJavaLibraryRule.class)
        .factoryClass(BazelJavaPlugin.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = java_plugin, TYPE = OTHER, FAMILY = Java) -->

<p>
  <code>java_plugin</code> defines plugins for the Java compiler run by Bazel. At the moment, the
  only supported kind of plugins are annotation processors. A <code>java_library</code> or
  <code>java_binary</code> rule can run plugins by depending on them via the <code>plugins</code>
  attribute. A <code>java_library</code> can also automatically export plugins to libraries that
  directly depend on it using
  <code><a href="#java_library.exported_plugins">exported_plugins</a></code>.
</p>

${IMPLICIT_OUTPUTS}

<p>
  Arguments are identical to <a href="#java_library"><code>java_library</code></a>, except for the
  addition of the <code>processor_class</code> argument.
</p>

<!-- #END_BLAZE_RULE -->*/
