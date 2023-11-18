// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import java.util.UUID;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for Skymeld with a dummy output service. */
@RunWith(JUnit4.class)
public class SkymeldOutputServiceBuildIntegrationTest extends BuildIntegrationTestCase {

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(
            new BlazeModule() {
              @Override
              public OutputService getOutputService() {
                // An output service that fails when #startBuild or #finalizeBuild is called.
                return new OutputService() {
                  @Override
                  public String getFilesSystemName() {
                    return "dummyTestFileSystem";
                  }

                  @Override
                  public ModifiedFileSet startBuild(
                      EventHandler eventHandler, UUID buildId, boolean finalizeActions) {
                    throw new IllegalStateException();
                  }

                  @Override
                  public void finalizeBuild(boolean buildSuccessful) {
                    throw new IllegalStateException();
                  }

                  @Override
                  public void finalizeAction(
                      Action action, OutputMetadataStore outputMetadataStore) {}

                  @Override
                  public BatchStat getBatchStatter() {
                    return null;
                  }

                  @Override
                  public boolean canCreateSymlinkTree() {
                    return false;
                  }

                  @Override
                  public void createSymlinkTree(
                      Map<PathFragment, PathFragment> symlinks, PathFragment symlinkTreeRoot) {}

                  @Override
                  public void clean() {}
                };
              }
            });
  }

  @Before
  public void setUp() {
    addOptions("--experimental_merged_skyframe_analysis_execution");
  }

  // Regression test for b/287277301.
  @Test
  public void noAnalyze_outputServiceStartBuildFinalizeBuildNotCalled() throws Exception {
    write(
        "foo/BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = ['foo.in'],",
        "  outs = ['foo.out'],",
        "  cmd = 'cp $< $@'",
        ")");
    write("foo/foo.in");
    addOptions("--noanalyze");

    BuildResult result = buildTarget("//foo:foo");

    assertThat(result.getSuccess()).isTrue();
  }
}
