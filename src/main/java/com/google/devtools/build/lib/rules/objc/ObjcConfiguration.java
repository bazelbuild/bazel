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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.starlarkbuildapi.apple.ObjcConfigurationApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** A compiler configuration containing flags required for Objective-C compilation. */
@Immutable
@RequiresOptions(options = {ObjcCommandLineOptions.class})
public class ObjcConfiguration extends Fragment implements ObjcConfigurationApi {
  private final CompilationMode compilationMode;
  private final boolean disallowSdkFrameworksAttributes;
  private final boolean alwayslinkByDefault;
  private final boolean stripExecutableSafely;
  private final boolean builtinObjcStripAction;
  private final boolean disableObjcFragment;

  public ObjcConfiguration(BuildOptions buildOptions) {
    CoreOptions options = buildOptions.get(CoreOptions.class);
    ObjcCommandLineOptions objcOptions = buildOptions.get(ObjcCommandLineOptions.class);

    this.compilationMode =
        Preconditions.checkNotNull(options.getCompilationMode(), "compilationMode");
    this.disallowSdkFrameworksAttributes =
        objcOptions.getIncompatibleDisallowSdkFrameworksAttributes();
    this.alwayslinkByDefault = objcOptions.getIncompatibleObjcAlwayslinkByDefault();
    this.stripExecutableSafely = objcOptions.getIncompatibleStripExecutableSafely();
    this.builtinObjcStripAction = objcOptions.getIncompatibleBuiltinObjcStripAction();
    this.disableObjcFragment = objcOptions.getDisableObjcFragment();
  }

  @Override
  public boolean shouldInclude() {
    return !disableObjcFragment;
  }

  @Override
  public ImmutableList<String> getCoptsForCompilationMode() {
    return switch (compilationMode) {
      case DBG, OPT -> ImmutableList.of();
      case FASTBUILD -> ImmutableList.of("-O0", "-DDEBUG=1");
      default -> throw new AssertionError();
    };
  }

  /** Returns whether sdk_frameworks and weak_sdk_frameworks attributes are disallowed. */
  @Override
  public boolean disallowSdkFrameworksAttributes() {
    return disallowSdkFrameworksAttributes;
  }

  /** Returns whether objc_library and objc_import should default to alwayslink=True. */
  @Override
  public boolean alwayslinkByDefault() {
    return alwayslinkByDefault;
  }

  /**
   * Looks at any explicit value for alwayslink on ctx and then falls back to the value of
   * alwayslink_by_default.
   */
  @Override
  public boolean targetShouldAlwayslink(StarlarkRuleContext ruleContext, StarlarkThread thread)
      throws EvalException {
    BuiltinRestriction.failIfCalledOutsideDefaultAllowlist(thread);

    AttributeMap attributes = ruleContext.getRuleContext().attributes();
    if (attributes.isAttributeValueExplicitlySpecified("alwayslink")) {
      return attributes.get("alwayslink", Type.BOOLEAN);
    }

    return alwayslinkByDefault;
  }

  /**
   * Returns whether executable strip action should use flag -x, which does not break dynamic symbol
   * resolution.
   */
  @Override
  public boolean stripExecutableSafely() {
    return stripExecutableSafely;
  }

  /** Returns whether to emit a strip action as part of objc linking. */
  @Override
  public boolean builtinObjcStripAction() {
    return builtinObjcStripAction;
  }
}
