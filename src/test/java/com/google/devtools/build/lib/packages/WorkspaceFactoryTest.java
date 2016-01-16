// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Package.LegacyBuilder;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.List;

/**
 * Tests for WorkspaceFactory.
 */
@RunWith(JUnit4.class)
public class WorkspaceFactoryTest {

  private Scratch scratch;
  private Path root;

  @Before
  public void setUpFileSystem() throws Exception {
    scratch = new Scratch("/");
    root = scratch.dir("/workspace");
  }

  @Test
  public void testLoadError() throws Exception {
    // WS with a syntax error: '//a' should end with .bzl.
    Path workspaceFilePath = scratch.file("/workspace/WORKSPACE", "load('//a', 'a')");
    LegacyBuilder builder = Package.newExternalPackageBuilder(workspaceFilePath, "");
    WorkspaceFactory factory =
        new WorkspaceFactory(
            builder,
            TestRuleClassProvider.getRuleClassProvider(),
            ImmutableList.<PackageFactory.EnvironmentExtension>of(),
            Mutability.create("test"),
            root,
            root);
    try {
      factory.parse(ParserInputSource.create(workspaceFilePath));
      fail("Parsing " + workspaceFilePath + " should have failed");
    } catch (IOException e) {
      assertThat(e.getMessage()).contains("Failed to parse " + workspaceFilePath);
    }
    List<Event> events = factory.getEvents();
    assertEquals(1, events.size());
    assertThat(events.get(0).getMessage())
        .contains("The label must reference a file with extension '.bzl'");
  }
}
