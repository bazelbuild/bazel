// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.filegroup;

import static com.google.devtools.build.lib.view.RunfilesProvider.RunfilesProviderImpl.dataSpecificRunfilesProvider;

import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.CompilationHelper;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.MiddlemanProvider;
import com.google.devtools.build.lib.view.MiddlemanProviderImpl;
import com.google.devtools.build.lib.view.RuleConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesCollector.State;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.view.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.view.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.view.test.InstrumentedFilesProviderImpl;

import java.util.Iterator;

/**
 * ConfiguredTarget for "filegroup".
 */
public class Filegroup implements RuleConfiguredTargetFactory {

  @Override
  public RuleConfiguredTarget create(RuleContext ruleContext) {
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.wrap(Order.STABLE_ORDER,
        ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET));
    NestedSet<Artifact> middleman = CompilationHelper.getAggregatingMiddleman(
        ruleContext, Actions.escapeLabel(ruleContext.getLabel()), filesToBuild);

    InstrumentedFilesCollector instrumentedFilesCollector =
        new InstrumentedFilesCollector(ruleContext,
            // what do *we* know about whether this is a source file or not
            new InstrumentationSpec(FileTypeSet.ANY_FILE, "srcs", "deps", "data"),
            InstrumentedFilesCollector.NO_METADATA_COLLECTOR);

    RunfilesProvider runfilesProvider = dataSpecificRunfilesProvider(
        new Runfiles.Builder().addRunfiles(State.DEFAULT, ruleContext).build(),
        // If you're visiting a filegroup as data, then we also visit its data as data.
        new Runfiles.Builder().addArtifacts(filesToBuild).addDataDeps(ruleContext).build());

    return new GenericRuleConfiguredTargetBuilder(ruleContext)
        .setRunfiles(runfilesProvider)
        .setFilesToBuild(filesToBuild)
        .setExecutable(getExecutable(filesToBuild))
        .add(InstrumentedFilesProvider.class, new InstrumentedFilesProviderImpl(
            instrumentedFilesCollector.getInstrumentedFiles(filesToBuild),
            instrumentedFilesCollector.getInstrumentationMetadataFiles(filesToBuild)))
        .add(MiddlemanProvider.class, new MiddlemanProviderImpl(middleman))
        .add(FilegroupPathProvider.class,
            new FilegroupPathProviderImpl(getFilegroupPath(ruleContext)))
        .build();
  }

  /*
   * Returns the single executable output of this filegroup. Returns
   * {@code null} if there are multiple outputs or the single output is not
   * considered an executable.
   */
  private Artifact getExecutable(NestedSet<Artifact> filesToBuild) {
    Iterator<Artifact> it = filesToBuild.iterator();
    if (it.hasNext()) {
      Artifact out = it.next();
      if (!it.hasNext()) {
        return out;
      }
    }
    return null;
  }

  private PathFragment getFilegroupPath(RuleContext ruleContext) {
    String attr = ruleContext.attributes().get("path", Type.STRING);
    if (attr.isEmpty()) {
      return PathFragment.EMPTY_FRAGMENT;
    } else {
      return ruleContext.getLabel().getPackageFragment().getRelative(attr);
    }
  }
}
