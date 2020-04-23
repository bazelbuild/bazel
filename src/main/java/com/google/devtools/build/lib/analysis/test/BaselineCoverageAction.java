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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifacts;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;

/** Generates baseline (empty) coverage for the given non-test target. */
@VisibleForTesting
@AutoCodec
@Immutable
public final class BaselineCoverageAction extends AbstractFileWriteAction
    implements NotifyOnActionCacheHit {
  private final NestedSet<Artifact> instrumentedFiles;

  @VisibleForSerialization
  BaselineCoverageAction(
      ActionOwner owner, NestedSet<Artifact> instrumentedFiles, Artifact primaryOutput) {
    super(owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), primaryOutput, false);
    this.instrumentedFiles = instrumentedFiles;
  }

  @Override
  public String getMnemonic() {
    return "BaselineCoverage";
  }

  @Override
  public void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    // TODO(b/150305897): No UUID?
    // TODO(b/150308417): Sort?
    Artifacts.addToFingerprint(fp, instrumentedFiles.toList());
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        PrintWriter writer = new PrintWriter(out);
        for (Artifact file : instrumentedFiles.toList()) {
          writer.write("SF:" + file.getExecPathString() + "\n");
          writer.write("end_of_record\n");
        }
        writer.flush();
      }
    };
  }

  @Override
  protected void afterWrite(ActionExecutionContext actionExecutionContext) {
    notifyAboutBaselineCoverage(actionExecutionContext.getEventHandler());
  }

  @Override
  public void actionCacheHit(ActionCachedContext context) {
    notifyAboutBaselineCoverage(context.getEventHandler());
  }

  /** Notify interested parties about new baseline coverage data. */
  private void notifyAboutBaselineCoverage(ExtendedEventHandler eventHandler) {
    Artifact output = Iterables.getOnlyElement(getOutputs());
    String ownerString = Label.print(getOwner().getLabel());
    eventHandler.post(new BaselineCoverageResult(output, ownerString));
  }

  /**
   * Returns collection of baseline coverage artifacts associated with the given target.
   * Will always return 0 or 1 elements.
   */
  static NestedSet<Artifact> create(
      RuleContext ruleContext, NestedSet<Artifact> instrumentedFiles) {
    // Baseline coverage artifacts will still go into "testlogs" directory.
    Artifact coverageData = ruleContext.getPackageRelativeArtifact(
        PathFragment.create(ruleContext.getTarget().getName()).getChild("baseline_coverage.dat"),
        ruleContext.getConfiguration().getTestLogsDirectory(
            ruleContext.getRule().getRepository()));
    ruleContext.registerAction(new BaselineCoverageAction(
        ruleContext.getActionOwner(), instrumentedFiles, coverageData));
    return NestedSetBuilder.create(Order.STABLE_ORDER, coverageData);
  }
}
