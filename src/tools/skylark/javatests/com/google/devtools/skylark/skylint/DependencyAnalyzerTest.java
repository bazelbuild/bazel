// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.skylark.skylint;

import com.google.common.collect.ImmutableMap;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.skylark.skylint.DependencyAnalyzer.DependencyCollector;
import com.google.devtools.skylark.skylint.Linter.FileFacade;
import java.nio.charset.StandardCharsets;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.skylark.skylint.DependencyAnalyzer} */
@RunWith(JUnit4.class)
public class DependencyAnalyzerTest {
  private static final DependencyCollector<List<String>> functionCollector =
      new DependencyCollector<List<String>>() {
        @Override
        public List<String> initInfo(Path path) {
          return new ArrayList<>();
        }

        @Override
        public List<String> loadDependency(
            List<String> currentFileInfo,
            LoadStatement stmt,
            Path loadedPath,
            List<String> loadedFileInfo) {
          for (String name : loadedFileInfo) {
            currentFileInfo.add(loadedPath + " - " + name);
          }
          return currentFileInfo;
        }

        @Override
        public List<String> collectInfo(Path path, BuildFileAST ast, List<String> info) {
          for (Statement stmt : ast.getStatements()) {
            if (stmt instanceof FunctionDefStatement) {
              info.add(((FunctionDefStatement) stmt).getIdentifier().getName());
            }
          }
          return info;
        }
      };

  public static FileFacade toFileFacade(Map<String, String> files) {
    return path -> {
      String contents = files.get(path.toString());
      if (contents == null) {
        throw new NoSuchFileException(path.toString());
      }
      return contents.getBytes(StandardCharsets.ISO_8859_1);
    };
  }

  private static List<String> getFunctionNames(Map<String, String> files, String path) {
    return new DependencyAnalyzer<>(toFileFacade(files), functionCollector)
        .collectTransitiveInfo(Paths.get(path));
  }

  @Test
  public void externalRepositoryDependency() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/BUILD", "")
            .put("/test.bzl", "load('@some-repo//foo:bar.bzl', 'baz')")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl")).isEmpty();
  }

  @Test
  public void nonexistentDependency() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/BUILD", "")
            .put("/test.bzl", "load(':does_not_exist.bzl', 'baz')")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl")).isEmpty();
  }

  @Test
  public void samePackageDependency() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/BUILD", "")
            .put("/test.bzl", "load(':bar.bzl', 'baz')\nload('//:bar.bzl', 'baz')")
            .put("/bar.bzl", "def baz(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl"))
        .isEqualTo(Arrays.asList("/bar.bzl - baz", "/bar.bzl - baz"));
  }

  @Test
  public void samePackageDependencyWithoutBuildFile() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/test.bzl", "load(':bar.bzl', 'baz')")
            .put("/bar.bzl", "def baz(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl")).isEmpty();
  }

  @Test
  public void subpackageDependency() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/BUILD", "")
            .put("/test.bzl", "load('subpackage:foo.bzl', 'foo')")
            .put("/subpackage/BUILD", "")
            .put("/subpackage/foo.bzl", "def foo(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl"))
        .isEqualTo(Collections.singletonList("/subpackage/foo.bzl - foo"));
  }

  @Test
  public void dependencyInSiblingPackage() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/pkg1/BUILD", "")
            .put("/pkg1/test.bzl", "load('//pkg2:bar.bzl', 'bar')")
            .put("/pkg2/BUILD", "")
            .put("/pkg2/bar.bzl", "def bar(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/pkg1/test.bzl"))
        .isEqualTo(Collections.singletonList("/pkg2/bar.bzl - bar"));
  }

  @Test
  public void dependencyInSiblingPackageWithBuildDotBazelFile() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/pkg1/BUILD.bazel", "")
            .put("/pkg1/test.bzl", "load('//pkg2:bar.bzl', 'bar')")
            .put("/pkg2/BUILD.bazel", "")
            .put("/pkg2/bar.bzl", "def bar(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/pkg1/test.bzl"))
        .isEqualTo(Collections.singletonList("/pkg2/bar.bzl - bar"));
  }

  @Test
  public void dependencyLabelWithoutPrefix() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/test/test.bzl", "load('bar.bzl', 'baz')")
            .put("/bar.bzl", "def baz(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test/test.bzl"))
        .isEqualTo(Collections.singletonList("/bar.bzl - baz"));
  }

  @Test
  public void transitiveDependencies() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/BUILD", "")
            .put("/test.bzl", "load(':foo.bzl', 'foo')")
            .put("/foo.bzl", "load(':bar.bzl', foo = 'bar')")
            .put("/bar.bzl", "def bar(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl"))
        .isEqualTo(Collections.singletonList("/foo.bzl - /bar.bzl - bar"));
  }

  @Test
  public void diamondDependencies() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/BUILD", "")
            .put("/test.bzl", "load(':foo.bzl', 'foo')\nload(':bar.bzl', 'bar')")
            .put("/foo.bzl", "load(':base.bzl', foo = 'base')")
            .put("/bar.bzl", "load(':base.bzl', bar = 'base')")
            .put("/base.bzl", "def base(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl"))
        .isEqualTo(Arrays.asList("/foo.bzl - /base.bzl - base", "/bar.bzl - /base.bzl - base"));
  }

  @Test
  public void cyclicDependenciesAreHandledGracefully() throws Exception {
    Map<String, String> files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/BUILD", "")
            .put("/test.bzl", "load(':test.bzl', 'foo')\ndef test(): pass")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl"))
        .isEqualTo(Collections.singletonList("test"));

    files =
        ImmutableMap.<String, String>builder()
            .put("/WORKSPACE", "")
            .put("/BUILD", "")
            .put("/test.bzl", "load(':foo.bzl', 'baz')\ndef test(): pass")
            .put("/foo.bzl", "load(':bar.bzl', 'baz')")
            .put("/bar.bzl", "load(':foo.bzl', 'baz')")
            .build();
    Truth.assertThat(getFunctionNames(files, "/test.bzl"))
        .isEqualTo(Collections.singletonList("test"));
  }
}
