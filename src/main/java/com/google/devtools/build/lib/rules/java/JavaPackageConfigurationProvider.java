// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static net.starlark.java.eval.Module.ofInnermostEnclosingStarlarkFunction;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.PackageSpecificationProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** A provider for Java per-package configuration. */
@AutoCodec
@AutoValue
@Immutable
public abstract class JavaPackageConfigurationProvider
    implements TransitiveInfoProvider, StarlarkValue {

  /** Creates a {@link JavaPackageConfigurationProvider}. */
  @AutoCodec.Instantiator
  public static JavaPackageConfigurationProvider create(
      ImmutableList<PackageSpecificationProvider> packageSpecifications,
      ImmutableList<String> javacopts,
      NestedSet<Artifact> data) {
    return new AutoValue_JavaPackageConfigurationProvider(packageSpecifications, javacopts, data);
  }

  /** Package specifications for which the configuration should be applied. */
  abstract ImmutableList<PackageSpecificationProvider> packageSpecifications();

  /** The javacopts for this configuration. */
  abstract ImmutableList<String> javacopts();

  abstract NestedSet<Artifact> data();

  /**
   * Returns true if this configuration matches the current label: that is, if the label's package
   * is contained by any of the {@link #packageSpecifications}.
   */
  public boolean matches(Label label) {
    // Do not use streams here as they create excessive garbage.
    for (PackageSpecificationProvider provider : packageSpecifications()) {
      for (PackageGroupContents specifications : provider.getPackageSpecifications().toList()) {
        if (specifications.containsPackage(label.getPackageIdentifier())) {
          return true;
        }
      }
    }
    return false;
  }

  @StarlarkMethod(
      name = "matches",
      documented = false,
      parameters = {
        @Param(
            name = "label",
            allowedTypes = {@ParamType(type = Label.class)})
      },
      useStarlarkThread = true)
  public boolean starlarkMatches(Label label, StarlarkThread starlarkThread) throws EvalException {
    validateAccess(starlarkThread);
    return matches(label);
  }

  @StarlarkMethod(name = "javac_opts", documented = false, useStarlarkThread = true)
  public Sequence<String> starlarkJavacOpts(StarlarkThread starlarkThread) throws EvalException {
    validateAccess(starlarkThread);
    return StarlarkList.immutableCopyOf(javacopts());
  }

  @StarlarkMethod(name = "data", documented = false, useStarlarkThread = true)
  public Depset starlarkData(StarlarkThread starlarkThread) throws EvalException {
    validateAccess(starlarkThread);
    return Depset.of(Artifact.TYPE, data());
  }

  private static void validateAccess(StarlarkThread starlarkThread) throws EvalException {
    RepositoryName repository =
        BazelModuleContext.of(ofInnermostEnclosingStarlarkFunction(starlarkThread))
            .label()
            .getRepository();
    if (!"@_builtins".equals(repository.getName())) {
      throw Starlark.errorf("private API only for use by builtins");
    }
  }
}
