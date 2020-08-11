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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaInfo;

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
    return constructJcapFromAspectDeps(
        ruleContext, javaProtoLibraryAspectProviders, /* alwaysStrict= */ false);
  }

  public static JavaCompilationArgsProvider constructJcapFromAspectDeps(
      RuleContext ruleContext,
      Iterable<JavaProtoLibraryAspectProvider> javaProtoLibraryAspectProviders,
      boolean alwaysStrict) {
    JavaCompilationArgsProvider strictCompProvider =
        JavaCompilationArgsProvider.merge(
            ruleContext.getPrerequisites(
                "deps", TransitionMode.TARGET, JavaCompilationArgsProvider.class));
    if (alwaysStrict || StrictDepsUtils.isStrictDepsJavaProtoLibrary(ruleContext)) {
      return strictCompProvider;
    } else {
      JavaCompilationArgsProvider.Builder nonStrictDirectJars =
          JavaCompilationArgsProvider.builder();
      for (JavaProtoLibraryAspectProvider p : javaProtoLibraryAspectProviders) {
        JavaCompilationArgsProvider args = p.getNonStrictCompArgs();
        nonStrictDirectJars
            .addRuntimeJars(args.getRuntimeJars())
            .addDirectCompileTimeJars(
                /* interfaceJars= */ args.getDirectCompileTimeJars(),
                /* fullJars= */ args.getDirectFullCompileTimeJars())
            .addTransitiveCompileTimeJars(args.getTransitiveCompileTimeJars());
      }
      // Don't collect .jdeps recursively for legacy "feature" compatibility reasons. Collecting
      // .jdeps here is probably a mistake; see JavaCompilationArgsProvider#makeNonStrict.
      return nonStrictDirectJars
          .addCompileTimeJavaDependencyArtifacts(
              strictCompProvider.getCompileTimeJavaDependencyArtifacts())
          .build();
    }
  }

  /**
   * Creates a JavaCompilationArgsProvider that's used when java_proto_library sets strict_deps=0.
   * It contains the jars a proto_library (or the proto aspect) produced, as well as all transitive
   * proto jars, and the proto runtime jars, all described as direct dependencies.
   */
  public static JavaCompilationArgsProvider createNonStrictCompilationArgsProvider(
      Iterable<JavaProtoLibraryAspectProvider> deps,
      JavaCompilationArgsProvider directJars,
      ImmutableList<TransitiveInfoCollection> protoRuntimes) {
    JavaCompilationArgsProvider.Builder result = JavaCompilationArgsProvider.builder();
    result.addExports(directJars);
    for (JavaProtoLibraryAspectProvider p : deps) {
      result.addExports(p.getNonStrictCompArgs());
    }
    for (TransitiveInfoCollection t : protoRuntimes) {
      JavaCompilationArgsProvider p = JavaInfo.getProvider(JavaCompilationArgsProvider.class, t);
      if (p != null) {
        result.addExports(p);
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
    if (ruleContext.getFragment(JavaConfiguration.class).strictDepsJavaProtos()
        || !ruleContext.attributes().has("strict_deps", Type.BOOLEAN)) {
      return true;
    }
    return (boolean) ruleContext.getRule().getAttr("strict_deps");
  }
}
