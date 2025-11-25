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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuiltins;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * Provides the binary artifact and its associated .dwp files, if fission is enabled. If Fission
 * ({@url https://gcc.gnu.org/wiki/DebugFission}) is not enabled, the dwp file will be null.
 */
public final class DebugPackageProvider {
  public static final Provider PROVIDER = new Provider();
  public static final RulesCcProvider RULES_CC_PROVIDER = new RulesCcProvider();

  public static DebugPackageProvider get(TransitiveInfoCollection target)
      throws RuleErrorException {
    DebugPackageProvider debugPackageProvider = target.get(PROVIDER);
    if (debugPackageProvider == null) {
      debugPackageProvider = target.get(RULES_CC_PROVIDER);
    }
    return debugPackageProvider;
  }

  private final StarlarkInfo starlarkInfo;

  private DebugPackageProvider(StarlarkInfo starlarkInfo) {
    this.starlarkInfo = starlarkInfo;
  }

  /** Returns the label for the *_binary target. */
  public final Label getTargetLabel() throws RuleErrorException {
    try {
      return starlarkInfo.getValue("target_label", Label.class);
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
  }

  /** Returns the stripped file (the explicit ".stripped" target). */
  public final Artifact getStrippedArtifact() throws RuleErrorException {
    try {
      return starlarkInfo.getNoneableValue("stripped_file", Artifact.class);
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
  }

  /** Returns the unstripped file (the default executable target). */
  public final Artifact getUnstrippedArtifact() throws RuleErrorException {
    try {
      return starlarkInfo.getValue("unstripped_file", Artifact.class);
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
  }

  /** Returns the .dwp file (for fission builds) or null if --fission=no. */
  @Nullable
  public final Artifact getDwpArtifact() throws RuleErrorException {
    try {
      return starlarkInfo.getNoneableValue("dwp_file", Artifact.class);
    } catch (EvalException e) {
      throw new RuleErrorException(e);
    }
  }

  /** Provider class for {@link DebugPackageProvider} objects. */
  public static class Provider extends StarlarkProviderWrapper<DebugPackageProvider> {
    public Provider() {
      super(
          keyForBuiltins(
              Label.parseCanonicalUnchecked("@_builtins//:common/cc/debug_package_info.bzl")),
          "DebugPackageInfo");
    }

    @Override
    public DebugPackageProvider wrap(Info value) throws RuleErrorException {
      return new DebugPackageProvider((StarlarkInfo) value);
    }
  }

  /** Provider class for {@link DebugPackageProvider} objects. */
  public static class RulesCcProvider extends StarlarkProviderWrapper<DebugPackageProvider> {
    public RulesCcProvider() {
      super(
          keyForBuild(
              Label.parseCanonicalUnchecked(
                  CppSemantics.RULES_CC_PREFIX + "cc/private:debug_package_info.bzl")),
          "DebugPackageInfo");
    }

    @Override
    public DebugPackageProvider wrap(Info value) throws RuleErrorException {
      return new DebugPackageProvider((StarlarkInfo) value);
    }
  }
}
