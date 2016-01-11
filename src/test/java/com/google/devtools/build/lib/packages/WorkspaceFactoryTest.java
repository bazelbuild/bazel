package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

/**
 * Tests for WorkspaceFactory.
 */
@RunWith(JUnit4.class)
public class WorkspaceFactoryTest  {

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
    try {
      parse(workspaceFilePath);
      fail("Parsing the WORKSPACE file should have failed.");
    } catch (IOException e) {
      assertThat(e.getMessage())
          .contains("The label must reference a file with extension '.bzl'");
    }
  }

  private Package.LegacyBuilder parse(Path workspaceFilePath) throws Exception {
    Package.LegacyBuilder builder = Package.newExternalPackageBuilder(workspaceFilePath, "");
    WorkspaceFactory factory = new WorkspaceFactory(
        builder, TestRuleClassProvider.getRuleClassProvider(), ImmutableList.of(),
        Mutability.create("test"), root, root);
    factory.parse(ParserInputSource.create(workspaceFilePath));
    return builder;
  }
}
