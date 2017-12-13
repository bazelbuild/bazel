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
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
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
  private Path workspaceFile;
  private SkylarkRepositoryContext context;

  @Before
  public void setUp() throws Exception {
    scratch = new Scratch("/");
    outputDirectory = scratch.dir("/outputDir");
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
    Package.Builder packageBuilder = Package.newExternalPackageBuilder(
        Package.Builder.DefaultHelper.INSTANCE, workspaceFile, "runfiles");
    FuncallExpression ast =
        new FuncallExpression(new Identifier("test"), ImmutableList.<Passed>of());
    ast.setLocation(Location.BUILTIN);
    Rule rule =
        WorkspaceFactoryHelper.createAndAddRepositoryRule(
            packageBuilder, buildRuleClass(attributes), null, kwargs, ast);
    HttpDownloader downloader = Mockito.mock(HttpDownloader.class);
    context =
        new SkylarkRepositoryContext(
            rule,
            outputDirectory,
            Mockito.mock(SkyFunction.Environment.class),
            ImmutableMap.of("FOO", "BAR"),
            downloader, new HashMap<String, String>());
  }

  protected void setUpContexForRule(String name) throws Exception {
    setUpContextForRule(ImmutableMap.<String, Object>of("name", name));
  }

  @Test
  public void testAttr() throws Exception {
    setUpContextForRule(
        ImmutableMap.<String, Object>of("name", "test", "foo", "bar"),
        Attribute.attr("foo", Type.STRING).build());

    assertThat(context.getAttr().getKeys()).contains("foo");
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

    assertThat(context.which("anything")).isNull();
    assertThat(context.which("def")).isNull();
    assertThat(context.which("undef")).isNull();
    assertThat(context.which("true").toString()).isEqualTo("/bin/true");
    assertThat(context.which("false").toString()).isEqualTo("/path/sbin/false");
  }

  @Test
  public void testFile() throws Exception {
    setUpContexForRule("test");
    context.createFile(context.path("foobar"), "", true);
    context.createFile(context.path("foo/bar"), "foobar", true);
    context.createFile(context.path("bar/foo/bar"), "", true);

    testOutputFile(outputDirectory.getChild("foobar"), "");
    testOutputFile(outputDirectory.getRelative("foo/bar"), "foobar");
    testOutputFile(outputDirectory.getRelative("bar/foo/bar"), "");

    try {
      context.createFile(context.path("/absolute"), "", true);
      fail("Expected error on creating path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot write outside of the repository directory for path /absolute");
    }
    try {
      context.createFile(context.path("../somepath"), "", true);
      fail("Expected error on creating path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot write outside of the repository directory for path /somepath");
    }
    try {
      context.createFile(context.path("foo/../../somepath"), "", true);
      fail("Expected error on creating path outside of the repository directory");
    } catch (RepositoryFunctionException ex) {
      assertThat(ex)
          .hasCauseThat()
          .hasMessageThat()
          .isEqualTo("Cannot write outside of the repository directory for path /somepath");
    }
  }

  @Test
  public void testSymlink() throws Exception {
    setUpContexForRule("test");
    context.createFile(context.path("foo"), "foobar", true);

    context.symlink(context.path("foo"), context.path("bar"));
    testOutputFile(outputDirectory.getChild("bar"), "foobar");

    assertThat(context.path("bar").realpath()).isEqualTo(context.path("foo"));
  }

  private void testOutputFile(Path path, String content) throws IOException {
    assertThat(path.exists()).isTrue();
    assertThat(
            CharStreams.toString(
                new InputStreamReader(path.getInputStream(), StandardCharsets.UTF_8)))
        .isEqualTo(content);
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
