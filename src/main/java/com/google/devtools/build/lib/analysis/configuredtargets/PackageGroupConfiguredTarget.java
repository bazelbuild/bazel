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

package com.google.devtools.build.lib.analysis.configuredtargets;

import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.PackageSpecificationProvider;
import com.google.devtools.build.lib.analysis.TargetContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Provider;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/**
 * Dummy ConfiguredTarget for package groups. Contains no functionality, since package groups are
 * not really first-class Targets.
 */
@Immutable
public class PackageGroupConfiguredTarget extends AbstractConfiguredTarget
    implements PackageSpecificationProvider, Info {

  private final NestedSet<PackageGroupContents> packageSpecifications;

  public static final BuiltinProvider<PackageGroupConfiguredTarget> PROVIDER =
      new BuiltinProvider<>("PackageSpecificationInfo", PackageGroupConfiguredTarget.class) {};

  // TODO(b/200065655): Only builtins should depend on a PackageGroupConfiguredTarget.
  //  Allowlists should be migrated to a new rule type that isn't package_group. Do not expose this
  //  to pure Starlark.
  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
    if (provider == FileProvider.class) {
      return provider.cast(FileProvider.EMPTY); // can't fail
    } else {
      return super.getProvider(provider);
    }
  }

  public PackageGroupConfiguredTarget(
      ActionLookupKey actionLookupKey,
      NestedSet<PackageGroupContents> visibility,
      NestedSet<PackageGroupContents> packageSpecifications) {
    super(actionLookupKey, visibility);
    this.packageSpecifications = packageSpecifications;
  }

  public PackageGroupConfiguredTarget(
      ActionLookupKey actionLookupKey, TargetContext targetContext, PackageGroup packageGroup) {
    this(
        actionLookupKey,
        targetContext.getVisibility(),
        getPackageSpecifications(targetContext, packageGroup));
  }

  private static NestedSet<PackageGroupContents> getPackageSpecifications(
      TargetContext targetContext, PackageGroup packageGroup) {
    NestedSetBuilder<PackageGroupContents> builder = NestedSetBuilder.stableOrder();
    for (Label label : packageGroup.getIncludes()) {
      TransitiveInfoCollection include =
          targetContext.findDirectPrerequisite(
              label, Optional.ofNullable(targetContext.getConfiguration()));
      PackageSpecificationProvider provider =
          include == null ? null : include.get(PackageGroupConfiguredTarget.PROVIDER);
      if (provider == null) {
        targetContext
            .getAnalysisEnvironment()
            .getEventHandler()
            .handle(
                Event.error(
                    targetContext.getTarget().getLocation(),
                    String.format("Label '%s' does not refer to a package group", label)));
        continue;
      }

      builder.addTransitive(provider.getPackageSpecifications());
    }

    builder.add(packageGroup.getPackageSpecifications());
    return builder.build();
  }

  @Override
  public NestedSet<PackageGroupContents> getPackageSpecifications() {
    return packageSpecifications;
  }

  @Override
  @Nullable
  protected Info rawGetStarlarkProvider(Provider.Key providerKey) {
    if (providerKey.equals(PROVIDER.getKey())) {
      return this;
    }
    return null;
  }

  @Override
  @Nullable
  protected Object rawGetStarlarkProvider(String providerKey) {
    return null;
  }

  @StarlarkMethod(
      name = "isAvailableFor",
      documented = false,
      parameters = {
        @Param(
            name = "label",
            allowedTypes = {@ParamType(type = Label.class)})
      },
      useStarlarkThread = true)
  public boolean starlarkMatches(Label label, StarlarkThread thread) throws EvalException {
    BuiltinRestriction.failIfCalledOutsideBuiltins(thread);
    return Allowlist.isAvailableFor(getPackageSpecifications(), label);
  }
}
