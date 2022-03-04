// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.IntegerValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.StringSequence;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.StringValue;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Simple unit tests for {@link LibraryToLinkValue}. */
@RunWith(JUnit4.class)
public class LibraryToLinkValueTest {
  @Test
  public void equalsAndHashCode() {
    EqualsTester equalsTester = new EqualsTester();

    // #forDynamicLibrary
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forDynamicLibrary("foo"), LibraryToLinkValue.forDynamicLibrary("foo"));
    equalsTester.addEqualityGroup(LibraryToLinkValue.forDynamicLibrary("bar"));

    // #forVersionedDynamicLibrary
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forVersionedDynamicLibrary("foo"),
        LibraryToLinkValue.forVersionedDynamicLibrary("foo"));
    equalsTester.addEqualityGroup(LibraryToLinkValue.forVersionedDynamicLibrary("bar"));

    // #forInterfaceLibrary
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forInterfaceLibrary("foo"),
        LibraryToLinkValue.forInterfaceLibrary("foo"));
    equalsTester.addEqualityGroup(LibraryToLinkValue.forInterfaceLibrary("bar"));

    // #forStaticLibrary
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forStaticLibrary("foo", /*isWholeArchive=*/ false),
        LibraryToLinkValue.forStaticLibrary("foo", /*isWholeArchive=*/ false));
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forStaticLibrary("foo", /*isWholeArchive=*/ true));
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forStaticLibrary("bar", /*isWholeArchive=*/ false));

    // #forObjectFile
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forObjectFile("foo", /*isWholeArchive=*/ false),
        LibraryToLinkValue.forObjectFile("foo", /*isWholeArchive=*/ false));
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forObjectFile("foo", /*isWholeArchive=*/ true));
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forObjectFile("bar", /*isWholeArchive=*/ false));

    ArtifactRoot sourceRoot =
        ArtifactRoot.asSourceRoot(
            Root.fromPath(
                new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/doesntmatter")));
    Artifact artifact1 =
        new SourceArtifact(sourceRoot, PathFragment.create("artifact1"), ArtifactOwner.NULL_OWNER);
    Artifact artifact2 =
        new SourceArtifact(sourceRoot, PathFragment.create("artifact2"), ArtifactOwner.NULL_OWNER);

    // #forObjectFileGroup
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forObjectFileGroup(
            ImmutableList.of(artifact1), /*isWholeArchive=*/ false),
        LibraryToLinkValue.forObjectFileGroup(
            ImmutableList.of(artifact1), /*isWholeArchive=*/ false));
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forObjectFileGroup(
            ImmutableList.of(artifact1), /*isWholeArchive=*/ true));
    equalsTester.addEqualityGroup(
        LibraryToLinkValue.forObjectFileGroup(
            ImmutableList.of(artifact2), /*isWholeArchive=*/ false));

    equalsTester.testEquals();
  }

  @Test
  public void getFieldValue_forDynamicLibrary() {
    LibraryToLinkValue libraryToLinkValue = LibraryToLinkValue.forDynamicLibrary("foo");
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("dynamic_library"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(0));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("foo"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
  }

  @Test
  public void getFieldValue_forVersionedDynamicLibrary() {
    LibraryToLinkValue libraryToLinkValue = LibraryToLinkValue.forVersionedDynamicLibrary("foo");
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("versioned_dynamic_library"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(0));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("foo"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
  }

  @Test
  public void getFieldValue_forInterfaceLibrary() {
    LibraryToLinkValue libraryToLinkValue = LibraryToLinkValue.forInterfaceLibrary("foo");
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("interface_library"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(0));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("foo"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
  }

  @Test
  public void getFieldValue_forStaticLibrary() {
    LibraryToLinkValue libraryToLinkValue =
        LibraryToLinkValue.forStaticLibrary("foo", /*isWholeArchive=*/ false);
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("static_library"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(0));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("foo"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
  }

  @Test
  public void getFieldValue_forStaticLibrary_isWholeArchive() {
    LibraryToLinkValue libraryToLinkValue =
        LibraryToLinkValue.forStaticLibrary("foo", /*isWholeArchive=*/ true);
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("static_library"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(1));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("foo"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
  }

  @Test
  public void getFieldValue_forObjectFile() {
    LibraryToLinkValue libraryToLinkValue =
        LibraryToLinkValue.forObjectFile("foo", /*isWholeArchive=*/ false);
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("object_file"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(0));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("foo"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
  }

  @Test
  public void getFieldValue_forObjectFile_isWholeArchive() {
    LibraryToLinkValue libraryToLinkValue =
        LibraryToLinkValue.forObjectFile("foo", /*isWholeArchive=*/ true);
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("object_file"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(1));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("foo"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
  }

  @Test
  public void getFieldValue_forObjectFileGroup() {
    Artifact artifact =
        new SourceArtifact(
            ArtifactRoot.asSourceRoot(
                Root.fromPath(new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/root"))),
            PathFragment.create("artifact"),
            ArtifactOwner.NULL_OWNER);

    LibraryToLinkValue libraryToLinkValue =
        LibraryToLinkValue.forObjectFileGroup(
            ImmutableList.of(artifact), /*isWholeArchive=*/ false);
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("object_file_group"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(0));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringSequence(ImmutableList.of("artifact")));
  }

  @Test
  public void getFieldValue_forObjectFileGroup_isWholeArchive() {
    Artifact artifact =
        new SourceArtifact(
            ArtifactRoot.asSourceRoot(
                Root.fromPath(new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/root"))),
            PathFragment.create("artifact"),
            ArtifactOwner.NULL_OWNER);

    LibraryToLinkValue libraryToLinkValue =
        LibraryToLinkValue.forObjectFileGroup(ImmutableList.of(artifact), /*isWholeArchive=*/ true);
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "type",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringValue("object_file_group"));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "is_whole_archive",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new IntegerValue(1));
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "name",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isNull();
    assertThat(
            libraryToLinkValue.getFieldValue(
                /*variableName=*/ "variable name doesn't matter",
                /*field=*/ "object_files",
                /*expander=*/ null,
                /*throwOnMissingVariable=*/ false))
        .isEqualTo(new StringSequence(ImmutableList.of("artifact")));
  }
}
