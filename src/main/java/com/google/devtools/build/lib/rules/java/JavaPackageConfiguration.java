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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.PackageSpecificationProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

/** Implementation for the java_package_configuration rule. */
public class JavaPackageConfiguration implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ImmutableList<PackageSpecificationProvider> packages =
        ImmutableList.copyOf(
            ruleContext.getPrerequisites("packages", PackageSpecificationProvider.class));
    FileProvider dataProvider = ruleContext.getPrerequisite("data", FileProvider.class);
    NestedSet<Artifact> data =
        dataProvider != null
            ? dataProvider.getFilesToBuild()
            : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    ImmutableList<String> javacopts =
        ImmutableList.copyOf(ruleContext.getExpander().withDataLocations().tokenized("javacopts"));
    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
        .setFilesToBuild(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
        .addProvider(JavaPackageConfigurationProvider.create(packages, javacopts, data))
        .build();
  }
}
