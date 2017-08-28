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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.WrappingProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;

public class StrictDepsUtils {

  /**
   * Used in JavaXXXProtoLibrary.java files to construct a JCAP from 'deps', where those were
   * populated by the Aspect it injected.
   *
   * <p>Takes care of strict deps.
   */
  public static JavaCompilationArgsProvider constructJcapFromAspectDeps(
      RuleContext ruleContext,
      Iterable<JavaProtoLibraryAspectProvider> javaProtoLibraryAspectProviders) {
    JavaCompilationArgsProvider strictCompProvider =
        JavaCompilationArgsProvider.merge(
            WrappingProvider.Helper.unwrapProviders(
                javaProtoLibraryAspectProviders, JavaCompilationArgsProvider.class));
    if (StrictDepsUtils.isStrictDepsJavaProtoLibrary(ruleContext)) {
      return strictCompProvider;
    } else {
      JavaCompilationArgs.Builder nonStrictDirectJars = JavaCompilationArgs.builder();
      for (JavaProtoLibraryAspectProvider p : javaProtoLibraryAspectProviders) {
        nonStrictDirectJars.addTransitiveArgs(p.getNonStrictCompArgs(), BOTH);
      }
      return JavaCompilationArgsProvider.create(
          nonStrictDirectJars.build(),
          strictCompProvider.getRecursiveJavaCompilationArgs(),
          strictCompProvider.getCompileTimeJavaDependencyArtifacts(),
          strictCompProvider.getRunTimeJavaDependencyArtifacts());
    }
  }

  /**
   * Creates a JavaCompilationArgsProvider that's used when java_proto_library sets strict_deps=0.
   * It contains the jars a proto_library (or the proto aspect) produced, as well as all transitive
   * proto jars, and the proto runtime jars, all described as direct dependencies.
   */
  public static JavaCompilationArgs createNonStrictCompilationArgsProvider(
      Iterable<JavaProtoLibraryAspectProvider> deps,
      JavaCompilationArgs directJars,
      ImmutableList<TransitiveInfoCollection> protoRuntimes) {
    JavaCompilationArgs.Builder result = JavaCompilationArgs.builder();
    for (JavaProtoLibraryAspectProvider p : deps) {
      result.addTransitiveArgs(p.getNonStrictCompArgs(), BOTH);
    }
    result.addTransitiveArgs(directJars, BOTH);
    for (TransitiveInfoCollection t : protoRuntimes) {
      JavaCompilationArgsProvider p = t.getProvider(JavaCompilationArgsProvider.class);
      if (p != null) {
        result.addTransitiveArgs(p.getJavaCompilationArgs(), BOTH);
      }
    }
    return result.build();
  }

  /**
   * Returns true iff 'ruleContext' should enforce strict-deps.
   *
   * <p>Using this method requires requesting the JavaConfiguration fragment.
   */
  public static boolean isStrictDepsJavaProtoLibrary(RuleContext ruleContext) {
    if (ruleContext.getFragment(JavaConfiguration.class).strictDepsJavaProtos()) {
      return true;
    }
    return (boolean) ruleContext.getRule().getAttributeContainer().getAttr("strict_deps");
  }
}
