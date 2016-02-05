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

package com.google.devtools.build.lib.skyframe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.rules.BazelRulesModule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.hamcrest.BaseMatcher;
import org.hamcrest.Description;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Matchers;
import org.mockito.Mockito;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

import java.io.IOException;

/**
 * Test for {@link WorkspaceFileFunction}.
 */
@RunWith(JUnit4.class)
public class WorkspaceFileFunctionTest extends BuildViewTestCase {

  private WorkspaceFileFunction workspaceSkyFunc;
  private ExternalPackageFunction externalSkyFunc;
  private WorkspaceASTFunction astSkyFunc;
  private FakeFileValue fakeWorkspaceFileValue;

  static class FakeFileValue extends FileValue {
    private boolean exists;
    private long size;

    FakeFileValue() {
      super();
      exists = true;
      size = 0L;
    }

    @Override
    public RootedPath realRootedPath() {
      throw new UnsupportedOperationException();
    }

    @Override
    public FileStateValue realFileStateValue() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean exists() {
      return exists;
    }

    void setExists(boolean exists) {
      this.exists = exists;
    }

    @Override
    public long getSize() {
      return size;
    }

    void setSize(long size) {
      this.size = size;
    }
  }

  @Before
  public final void setUp() throws Exception {
    ConfiguredRuleClassProvider ruleClassProvider = TestRuleClassProvider.getRuleClassProvider();
    workspaceSkyFunc =
        new WorkspaceFileFunction(
            ruleClassProvider,
            new PackageFactory(
                ruleClassProvider, new BazelRulesModule().getPackageEnvironmentExtension()),
            directories);
    externalSkyFunc = new ExternalPackageFunction();
    astSkyFunc = new WorkspaceASTFunction(ruleClassProvider);
    fakeWorkspaceFileValue = new FakeFileValue();
  }

  private Label getLabelMapping(Package pkg, String name) throws NoSuchTargetException {
    return (Label) ((Rule) pkg.getTarget(name)).getAttributeContainer().getAttr("actual");
  }

  private RootedPath createWorkspaceFile(String... contents) throws IOException {
    Path workspacePath = scratch.overwriteFile("WORKSPACE", contents);
    fakeWorkspaceFileValue.setSize(workspacePath.getFileSize());
    return RootedPath.toRootedPath(
        workspacePath.getParentDirectory(), new PathFragment(workspacePath.getBaseName()));
  }

  // Dummy harmcrest matcher that match the function name of a skykey
  static class SkyKeyMatchers extends BaseMatcher<SkyKey> {
    private final SkyFunctionName functionName;
    
    public SkyKeyMatchers(SkyFunctionName functionName) {
      this.functionName = functionName;
    }
    @Override
    public boolean matches(Object item) {
      if (item instanceof SkyKey) {
        return ((SkyKey) item).functionName().equals(functionName);
      }
      return false;
    }
    
    @Override
    public void describeTo(Description description) {}
  }

  private SkyFunction.Environment getEnv() {
    SkyFunction.Environment env = Mockito.mock(SkyFunction.Environment.class);
    Mockito.when(env.getValue(Matchers.argThat(new SkyKeyMatchers(SkyFunctions.FILE))))
        .thenReturn(fakeWorkspaceFileValue);
    Mockito.when(env.getValue(Matchers.argThat(new SkyKeyMatchers(SkyFunctions.WORKSPACE_FILE))))
        .then(
            new Answer<SkyValue>() {
              @Override
              public SkyValue answer(InvocationOnMock invocation) throws Throwable {
                SkyKey key = (SkyKey) invocation.getArguments()[0];
                return workspaceSkyFunc.compute(key, getEnv());
              }
            });
    Mockito.when(env.getValue(Matchers.argThat(new SkyKeyMatchers(SkyFunctions.WORKSPACE_AST))))
        .then(
            new Answer<SkyValue>() {
              @Override
              public SkyValue answer(InvocationOnMock invocation) throws Throwable {
                SkyKey key = (SkyKey) invocation.getArguments()[0];
                return astSkyFunc.compute(key, getEnv());
              }
            });
    return env;
  }

  @Test
  public void testInvalidRepo() throws Exception {
    RootedPath workspacePath = createWorkspaceFile("workspace(name = 'foo$')");
    PackageValue value =
        (PackageValue) externalSkyFunc
            .compute(ExternalPackageFunction.key(workspacePath), getEnv());
    Package pkg = value.getPackage();
    assertTrue(pkg.containsErrors());
    MoreAsserts.assertContainsEvent(pkg.getEvents(), "foo$ is not a legal workspace name");
  }

  @Test
  public void testBindFunction() throws Exception {
    String lines[] = {"bind(name = 'foo/bar',", "actual = '//foo:bar')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key(workspacePath);
    PackageValue value = (PackageValue) externalSkyFunc.compute(key, getEnv());
    Package pkg = value.getPackage();
    assertEquals(Label.parseAbsolute("//foo:bar"), getLabelMapping(pkg, "foo/bar"));
    MoreAsserts.assertNoEvents(pkg.getEvents());
  }

  @Test
  public void testBindArgsReversed() throws Exception {
    String lines[] = {"bind(actual = '//foo:bar', name = 'foo/bar')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key(workspacePath);
    PackageValue value = (PackageValue) externalSkyFunc.compute(key, getEnv());
    Package pkg = value.getPackage();
    assertEquals(Label.parseAbsolute("//foo:bar"), getLabelMapping(pkg, "foo/bar"));
    MoreAsserts.assertNoEvents(pkg.getEvents());
  }

  @Test
  public void testNonExternalBinding() throws Exception {
    // name must be a valid label name.
    String lines[] = {"bind(name = 'foo:bar', actual = '//bar/baz')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    PackageValue value =
        (PackageValue) externalSkyFunc
            .compute(ExternalPackageFunction.key(workspacePath), getEnv());
    Package pkg = value.getPackage();
    assertTrue(pkg.containsErrors());
    MoreAsserts.assertContainsEvent(pkg.getEvents(), "target names may not contain ':'");
  }

  @Test
  public void testWorkspaceFileParsingError() throws Exception {
    // //external:bar:baz is not a legal package.
    String lines[] = {"bind(name = 'foo/bar', actual = '//external:bar:baz')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    PackageValue value =
        (PackageValue) externalSkyFunc
            .compute(ExternalPackageFunction.key(workspacePath), getEnv());
    Package pkg = value.getPackage();
    assertTrue(pkg.containsErrors());
    MoreAsserts.assertContainsEvent(pkg.getEvents(), "target names may not contain ':'");
  }

  @Test
  public void testNoWorkspaceFile() throws Exception {
    // Even though the WORKSPACE exists, Skyframe thinks it doesn't, so it doesn't.
    String lines[] = {"bind(name = 'foo/bar', actual = '//foo:bar')"};
    RootedPath workspacePath = createWorkspaceFile(lines);
    fakeWorkspaceFileValue.setExists(false);

    PackageValue value =
        (PackageValue) externalSkyFunc
            .compute(ExternalPackageFunction.key(workspacePath), getEnv());
    Package pkg = value.getPackage();
    assertFalse(pkg.containsErrors());
    MoreAsserts.assertNoEvents(pkg.getEvents());
  }

  @Test
  public void testListBindFunction() throws Exception {
    String lines[] = {
        "L = ['foo', 'bar']", "bind(name = '%s/%s' % (L[0], L[1]),", "actual = '//foo:bar')"};
    RootedPath workspacePath = createWorkspaceFile(lines);

    SkyKey key = ExternalPackageFunction.key(workspacePath);
    PackageValue value = (PackageValue) externalSkyFunc.compute(key, getEnv());
    Package pkg = value.getPackage();
    assertEquals(Label.parseAbsolute("//foo:bar"), getLabelMapping(pkg, "foo/bar"));
    MoreAsserts.assertNoEvents(pkg.getEvents());
  }
}
