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

package com.google.devtools.build.lib.rules.java.proto;

import static com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType.BOTH;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;

public class StrictDepsUtils {

  /**
   * Returns true iff 'ruleContext' should enforce strict-deps.
   *
   * <ol>
   * <li>If the rule explicitly specifies the 'strict_deps' attribute, returns its value.
   * <li>Otherwise, if the package explicitly specifies 'default_strict_deps_java_proto_library',
   *     returns that value.
   * <li>Otherwise, returns the value of the --strict_deps_java_proto_library flag.
   * </ol>
   *
   * Using this method requires requesting the JavaConfiguration fragment.
   */
  public static boolean isStrictDepsJavaProtoLibrary(RuleContext ruleContext) {
    AttributeContainer attributeContainer = ruleContext.getRule().getAttributeContainer();
    if (attributeContainer.isAttributeValueExplicitlySpecified("strict_deps")) {
      return (boolean) attributeContainer.getAttr("strict_deps");
    }
    TriState defaultJavaProtoLibraryStrictDeps =
        ruleContext.getRule().getPackage().getDefaultStrictDepsJavaProtos();
    if (defaultJavaProtoLibraryStrictDeps == TriState.AUTO) {
      return ruleContext.getFragment(JavaConfiguration.class).strictDepsJavaProtos();
    }
    return defaultJavaProtoLibraryStrictDeps == TriState.YES;
  }

  /**
   * Returns a new JavaCompilationArgsProvider whose direct-jars part is the union of both the
   * direct and indirect jars of 'provider'.
   */
  public static JavaCompilationArgsProvider makeNonStrict(JavaCompilationArgsProvider provider) {
    JavaCompilationArgs.Builder directCompilationArgs = JavaCompilationArgs.builder();
    directCompilationArgs
        .addTransitiveArgs(provider.getJavaCompilationArgs(), BOTH)
        .addTransitiveArgs(provider.getRecursiveJavaCompilationArgs(), BOTH);
    return JavaCompilationArgsProvider.create(
        directCompilationArgs.build(),
        provider.getRecursiveJavaCompilationArgs(),
        provider.getCompileTimeJavaDependencyArtifacts(),
        provider.getRunTimeJavaDependencyArtifacts());
  }
}
