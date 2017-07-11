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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package.Builder;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for WorkspaceFactory.
 */
@RunWith(JUnit4.class)
public class WorkspaceFactoryTest {

  @Test
  public void testLoadError() throws Exception {
    // WS with a syntax error: '//a' should end with .bzl.
    WorkspaceFactoryHelper helper = parse("load('//a', 'a')");
    helper.assertLexingExceptionThrown();
    assertThat(helper.getLexerError())
        .contains("The label must reference a file with extension '.bzl'");
  }

  @Test
  public void testWorkspaceName() throws Exception {
    WorkspaceFactoryHelper helper = parse("workspace(name = 'my_ws')");
    assertThat(helper.getPackage().getWorkspaceName()).isEqualTo("my_ws");
  }

  @Test
  public void testWorkspaceStartsWithNumber() throws Exception {
    WorkspaceFactoryHelper helper = parse("workspace(name = '123abc')");
    assertThat(helper.getParserError()).contains("123abc is not a legal workspace name");
  }

  @Test
  public void testWorkspaceWithIllegalCharacters() throws Exception {
    WorkspaceFactoryHelper helper = parse("workspace(name = 'a.b.c')");
    assertThat(helper.getParserError()).contains("a.b.c is not a legal workspace name");
  }

  @Test
  public void testIllegalRepoName() throws Exception {
    WorkspaceFactoryHelper helper = parse("local_repository(",
        "    name = 'foo/bar',",
        "    path = '/foo/bar',",
        ")");
    assertThat(helper.getParserError()).contains(
        "local_repository rule //external:foo/bar's name field must be a legal workspace name");
  }

  @Test
  public void testIllegalWorkspaceFunctionPosition() throws Exception {
    WorkspaceFactoryHelper helper = new WorkspaceFactoryHelper(
        false, "workspace(name = 'foo')");
    assertThat(helper.getParserError()).contains(
        "workspace() function should be used only at the top of the WORKSPACE file");
  }

  @Test
  public void testRegisterToolchains() throws Exception {
    WorkspaceFactoryHelper helper = parse("register_toolchains('//toolchain:tc1')");
    assertThat(helper.getPackage().getRegisteredToolchainLabels())
        .containsExactly(Label.parseAbsolute("//toolchain:tc1"));
  }

  @Test
  public void testRegisterToolchains_multipleLabels() throws Exception {
    WorkspaceFactoryHelper helper =
        parse("register_toolchains(", "  '//toolchain:tc1',", "  '//toolchain:tc2')");
    assertThat(helper.getPackage().getRegisteredToolchainLabels())
        .containsExactly(
            Label.parseAbsolute("//toolchain:tc1"), Label.parseAbsolute("//toolchain:tc2"));
  }

  @Test
  public void testRegisterToolchains_multipleCalls() throws Exception {
    WorkspaceFactoryHelper helper =
        parse("register_toolchains('//toolchain:tc1')", "register_toolchains('//toolchain:tc2')");
    assertThat(helper.getPackage().getRegisteredToolchainLabels())
        .containsExactly(
            Label.parseAbsolute("//toolchain:tc1"), Label.parseAbsolute("//toolchain:tc2"));
  }

  private WorkspaceFactoryHelper parse(String... args) {
    return new WorkspaceFactoryHelper(args);
  }

  /**
   * Parses a WORKSPACE file with the given content.
   */
  private class WorkspaceFactoryHelper {
    private final Builder builder;
    private final WorkspaceFactory factory;
    private final Exception exception;
    private final ImmutableList<Event> events;

    public WorkspaceFactoryHelper(String... args) {
      this(true, args);
    }

    public WorkspaceFactoryHelper(boolean allowOverride, String... args) {
      Path root = null;
      Path workspaceFilePath = null;
      try {
        Scratch scratch = new Scratch("/");
        root = scratch.dir("/workspace");
        workspaceFilePath = scratch.file("/workspace/WORKSPACE", args);
      } catch (IOException e) {
        fail("Shouldn't happen: " + e.getMessage());
      }
      StoredEventHandler eventHandler = new StoredEventHandler();
      builder = Package.newExternalPackageBuilder(
          Package.Builder.DefaultHelper.INSTANCE, workspaceFilePath, "");
      this.factory = new WorkspaceFactory(
          builder,
          TestRuleClassProvider.getRuleClassProvider(),
          ImmutableList.<PackageFactory.EnvironmentExtension>of(),
          Mutability.create("test"),
          allowOverride,
          root,
          root);
      Exception exception = null;
      try {
        factory.parse(ParserInputSource.create(workspaceFilePath), eventHandler);
      } catch (BuildFileContainsErrorsException e) {
        exception = e;
      } catch (IOException | InterruptedException e) {
        fail("Shouldn't happen: " + e.getMessage());
      }
      this.events = eventHandler.getEvents();
      this.exception = exception;
    }

    public Package getPackage() throws InterruptedException {
      return builder.build();
    }

    public void assertLexingExceptionThrown() {
      assertThat(exception).isNotNull();
      assertThat(exception).hasMessageThat().contains("Failed to parse /workspace/WORKSPACE");
    }

    public String getLexerError() {
      assertThat(events).hasSize(1);
      return events.get(0).getMessage();
    }

    public String getParserError() {
      List<Event> events = builder.getEvents();
      assertThat(events.size()).isGreaterThan(0);
      return events.get(0).getMessage();
    }
  }
}
