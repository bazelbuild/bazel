// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.runtime.events.GotProjectFileEvent;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ProjectFileSupport}. */
@RunWith(JUnit4.class)
public class ProjectFileSupportTest {

  /** An implementation of ProjectFile that always returns a single target. */
  private static class DummyProjectFile implements ProjectFile {

    private final PathFragment path;

    public DummyProjectFile(PathFragment path) {
      this.path = path;
    }

    @Override
    public String getName() {
      return path.toString();
    }

    @Override
    public List<String> getCommandLineFor(String command, EventHandler eventHandler) {
      return ImmutableList.of("//dummy:target");
    }
  }

  private static class MockProjectFileProvider implements ProjectFile.Provider {
    @Override
    public ProjectFile getProjectFile(
        Path workingDirectory,
        List<Root> packagePath,
        PathFragment path,
        OptionsParser optionsParser) {
      return new DummyProjectFile(path);
    }
  }

  private final StoredEventHandler events = new StoredEventHandler();
  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);

  private static OptionsParser createOptionsParser() {
    return OptionsParser.builder()
        .optionsClasses(CommonCommandOptions.class, PackageOptions.class)
        .allowResidue(true)
        .build();
  }

  protected OptionsParser initializeOptionsParser() throws Exception {
    OptionsParser parser = createOptionsParser();
    parser.parse("--experimental_allow_project_files"); // allow this test to work
    return parser;
  }

  private void handleProjectFiles(OptionsParser optionsParser, String command) throws Exception {
    Path workspacePath = fs.getPath("/workspace");
    workspacePath.createDirectoryAndParents();
    ProjectFileSupport.handleProjectFiles(
        events,
        new MockProjectFileProvider(),
        workspacePath.asFragment(),
        workspacePath,
        optionsParser,
        command);
  }

  @Test
  public void testGetTargets_noProjectFile() throws Exception {
    OptionsParser optionsParser = initializeOptionsParser();
    optionsParser.parse("//tar:get", "//oth:er");

    List<String> targets =
        ProjectFileSupport.getTargets(new MockProjectFileProvider(), optionsParser);
    assertThat(targets).containsExactly("//tar:get", "//oth:er").inOrder();
  }

  @Test
  public void testGetTargets_withProjectFile() throws Exception {
    OptionsParser optionsParser = initializeOptionsParser();
    optionsParser.parse("+my.proj:unit");

    List<String> targets =
        ProjectFileSupport.getTargets(new MockProjectFileProvider(), optionsParser);
    assertThat(targets).isEmpty();
  }

  @Test
  public void testGetTargets_withProjectFileAndTargets() throws Exception {
    OptionsParser optionsParser = initializeOptionsParser();
    optionsParser.parse("+my.proj:unit", "//other:stuff");

    List<String> targets =
        ProjectFileSupport.getTargets(new MockProjectFileProvider(), optionsParser);
    assertThat(targets).containsExactly("//other:stuff");
  }

  @Test
  public void testHandleProjectFiles_noProjectFile() throws Exception {
    OptionsParser optionsParser = initializeOptionsParser();
    optionsParser.parse("//tar:get");

    handleProjectFiles(optionsParser, "build");
    assertThat(optionsParser.getResidue()).containsExactly("//tar:get");
  }

  @Test
  public void testHandleProjectFiles_twoProjectFiles() throws Exception {
    OptionsParser optionsParser = initializeOptionsParser();
    optionsParser.parse("+this.file", "+that.file");

    assertThrows(OptionsParsingException.class, () -> handleProjectFiles(optionsParser, "build"));
  }

  @Test
  public void testHandleProjectFiles_projectFilesDisabled() throws Exception {
    OptionsParser optionsParser = initializeOptionsParser();
    optionsParser.parse("--noexperimental_allow_project_files", "+this.file");

    assertThrows(OptionsParsingException.class, () -> handleProjectFiles(optionsParser, "build"));
  }

  @Test
  public void testHandleProjectFiles_populatesOptions() throws Exception {
    OptionsParser optionsParser = initializeOptionsParser();
    optionsParser.parse("+project.file");

    handleProjectFiles(optionsParser, "build");
    assertThat(optionsParser.getResidue())
        .containsExactly("+project.file", "//dummy:target")
        .inOrder();
  }

  @Test
  public void testHandleProjectFiles_postsGotProjectFileEvent() throws Exception {
    OptionsParser optionsParser = initializeOptionsParser();
    optionsParser.parse("+project.file");

    handleProjectFiles(optionsParser, "build");
    assertThat(events.getPosts()).hasSize(1);
    Object post = events.getPosts().get(0);
    assertThat(post).isInstanceOf(GotProjectFileEvent.class);
  }
}
