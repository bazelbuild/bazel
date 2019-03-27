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

package com.google.devtools.build.lib.bazel.repository.skylark;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.WorkspaceFactoryHelper;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.syntax.Argument.Passed;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/**
 * Unit tests for complex function of SkylarkRepositoryContext.
 */
@RunWith(JUnit4.class)
public class SkylarkRepositoryContextTest {

  private Scratch scratch;
  private Path outputDirectory;
  private Root root;
  private Path workspaceFile;
  private SkylarkRepositoryContext context;

  @Before
  public void setUp() throws Exception {
    scratch = new Scratch("/");
    outputDirectory = scratch.dir("/outputDir");
    root = Root.fromPath(scratch.dir("/"));
    workspaceFile = scratch.file("/WORKSPACE");
  }

  protected static RuleClass buildRuleClass(Attribute... attributes) {
    RuleClass.Builder ruleClassBuilder =
        new RuleClass.Builder("test", RuleClassType.WORKSPACE, true);
    for (Attribute attr : attributes) {
      ruleClassBuilder.addOrOverrideAttribute(attr);
    }
    ruleClassBuilder.setWorkspaceOnly();
    ruleClassBuilder.setConfiguredTargetFunction(new BuiltinFunction("test") {});
    return ruleClassBuilder.build();
  }

  protected void setUpContextForRule(Map<String, Object> kwargs, Attribute... attributes)
      throws Exception {
    Package.Builder packageBuilder =
        Package.newExternalPackageBuilder(
            Package.Builder.DefaultHelper.INSTANCE,
            RootedPath.toRootedPath(root, workspaceFile),
            "runfiles");
    FuncallExpression ast =
        new FuncallExpression(Identifier.of("test"), ImmutableList.<Passed>of());
    ast.setLocation(Location.BUILTIN);
    Rule rule =
        WorkspaceFactoryHelper.createAndAddRepositoryRule(
            packageBuilder, buildRuleClass(attributes), null, kwargs, ast);
    HttpDownloader downloader = Mockito.mock(HttpDownloader.class);
    SkyFunction.Environment environment = Mockito.mock(SkyFunction.Environment.class);
    ExtendedEventHandler listener = Mockito.mock(ExtendedEventHandler.class);
    Mockito.when(environment.getListener()).thenReturn(listener);
    context =
        new SkylarkRepositoryContext(
            rule,
            outputDirectory,
            environment,
            ImmutableMap.of("FOO", "BAR"),
            downloader,
            1.0,
            new HashMap<String, String>());
  }

  protected void setUpContexForRule(String name) throws Exception {
    setUpContextForRule(ImmutableMap.<String, Object>of("name", name));
  }

  @Test
  public void testAttr() throws Exception {
    setUpContextForRule(
        ImmutableMap.<String, Object>of("name", "test", "foo", "bar"),
        Attribute.attr("foo", Type.STRING).build());

    assertThat(context.getAttr().getFieldNames()).contains("foo");
    assertThat(context.getAttr().getValue("foo")).isEqualTo("bar");
  }

  @Test
  public void testWhich() throws Exception {
    setUpContexForRule("test");
    SkylarkRepositoryContext.setPathEnvironment("/bin", "/path/sbin", ".");
    scratch.file("/bin/true").setExecutable(true);
    scratch.file("/path/sbin/true").setExecutable(true);
    scratch.file("/path/sbin/false").setExecutable(true);
    scratch.file("/path/bin/undef").setExecutable(true);
    scratch.file("/path/bin/def").setExecutable(true);
    scratch.file("/bin/undef");

    assertThat(context.which("anything", null)).isNull();
    assertThat(context.which("def", null)).isNull();
    assertThat(context.which("undef", null)).isNull();
    assertThat(context.which("true", null).toString()).isEqualTo("/bin/true");
    assertThat(context.which("false", null).toString()).isEqualTo("/path/sbin/false");
  }

  @Test
  public void testFile() throws Exception {
    setUpContexForRule("test");
    context.createFile(context.path("foobar"), "", true, true, null);
    context.createFile(context.path("foo/bar"), "foobar", true, true, null);
    context.createFile(context.path("bar/foo/bar"), "", true, true, null);

    testOutputFile(outputDirectory.getChild("foobar"), "");
    testOutputFile(outputDirectory.getRelative("foo/bar"), "foobar");
    testOutputFile(outputDirectory.getRelative("bar/foo/bar"), "");

    try {
      context.createFile(context.path("/absolute"), "", true, true, null);
      fail("Expected error on creating path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot write outside of the repository directory for path /absolute");
    }
    try {
      context.createFile(context.path("../somepath"), "", true, true, null);
      fail("Expected error on creating path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot write outside of the repository directory for path /somepath");
    }
    try {
      context.createFile(context.path("foo/../../somepath"), "", true, true, null);
      fail("Expected error on creating path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot write outside of the repository directory for path /somepath");
    }
  }

  @Test
  public void testDelete() throws Exception {
    setUpContexForRule("testDelete");
    context.createFile(context.path("foo/bar"), "content", true, true, null);
    assertThat(context.delete("foo/bar", null)).isTrue();

    assertThat(context.delete("abc/def", null)).isFalse();
    Path tempFile = scratch.file("/abcde/b", "123");
    assertThat(context.delete(context.path(tempFile.getPathString()), null)).isTrue();

    Path innerDir = scratch.dir("/some/inner");
    scratch.dir("/some/inner/deeper");
    scratch.file("/some/inner/deeper.txt");
    scratch.file("/some/inner/deeper/1.txt");
    assertThat(context.delete(innerDir.toString(), null)).isTrue();
  }

  @Test
  public void testRead() throws Exception {
    setUpContexForRule("test");
    context.createFile(context.path("foo/bar"), "foobar", true, true, null);

    String content = context.readFile(context.path("foo/bar"), null);
    assertThat(content).isEqualTo("foobar");

    try {
      context.readFile(context.path("/absolute"), null);
      fail("Expected error on reading path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot read outside of the repository directory for path /absolute");
    }
    try {
      context.readFile(context.path("../somepath"), null);
      fail("Expected error on reading path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot read outside of the repository directory for path /somepath");
    }
    try {
      context.readFile(context.path("foo/../../somepath"), null);
      fail("Expected error on reading path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot read outside of the repository directory for path /somepath");
    }
  }

  @Test
  public void testSymlink() throws Exception {
    setUpContexForRule("test");
    context.createFile(context.path("foo"), "foobar", true, true, null);

    context.symlink(context.path("foo"), context.path("bar"), null);
    testOutputFile(outputDirectory.getChild("bar"), "foobar");

    assertThat(context.path("bar").realpath()).isEqualTo(context.path("foo"));
  }

  private void testOutputFile(Path path, String content) throws IOException {
    assertThat(path.exists()).isTrue();
    try (InputStreamReader reader =
        new InputStreamReader(path.getInputStream(), StandardCharsets.UTF_8)) {
      assertThat(CharStreams.toString(reader)).isEqualTo(content);
    }
  }

  @Test
  public void testDirectoryListing() throws Exception {
    setUpContexForRule("test");
    scratch.file("/my/folder/a");
    scratch.file("/my/folder/b");
    scratch.file("/my/folder/c");
    assertThat(context.path("/my/folder").readdir()).containsExactly(
        context.path("/my/folder/a"), context.path("/my/folder/b"), context.path("/my/folder/c"));
  }
}
