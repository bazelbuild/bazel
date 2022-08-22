// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link IndexRegistry}. */
@RunWith(JUnit4.class)
public class IndexRegistryTest extends FoundationTestCase {
  @Rule public final TestHttpServer server = new TestHttpServer();
  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

  private RegistryFactory registryFactory;

  @Before
  public void setUp() throws Exception {
    registryFactory =
        new RegistryFactoryImpl(new HttpDownloader(), Suppliers.ofInstance(ImmutableMap.of()));
  }

  @Test
  public void testHttpUrl() throws Exception {
    server.serve("/myreg/modules/foo/1.0/MODULE.bazel", "lol");
    server.start();

    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl() + "/myreg");
    assertThat(registry.getModuleFile(createModuleKey("foo", "1.0"), reporter))
        .hasValue("lol".getBytes(UTF_8));
    assertThat(registry.getModuleFile(createModuleKey("bar", "1.0"), reporter)).isEmpty();
  }

  @Test
  public void testFileUrl() throws Exception {
    tempFolder.newFolder("fakereg", "modules", "foo", "1.0");
    File file = tempFolder.newFile("fakereg/modules/foo/1.0/MODULE.bazel");
    try (Writer writer = Files.newBufferedWriter(file.toPath(), UTF_8)) {
      writer.write("lol");
    }

    Registry registry =
        registryFactory.getRegistryWithUrl(
            new File(tempFolder.getRoot(), "fakereg").toURI().toString());
    assertThat(registry.getModuleFile(createModuleKey("foo", "1.0"), reporter))
        .hasValue("lol".getBytes(UTF_8));
    assertThat(registry.getModuleFile(createModuleKey("bar", "1.0"), reporter)).isEmpty();
  }

  @Test
  public void testGetRepoSpec() throws Exception {
    server.serve(
        "/bazel_registry.json",
        "{",
        "  \"mirrors\": [",
        "    \"https://mirror.bazel.build/\",",
        "    \"file:///home/bazel/mymirror/\"",
        "  ]",
        "}");
    server.serve(
        "/modules/foo/1.0/source.json",
        "{",
        "  \"url\": \"http://mysite.com/thing.zip\",",
        "  \"integrity\": \"sha256-blah\",",
        "  \"strip_prefix\": \"pref\"",
        "}");
    server.serve(
        "/modules/bar/2.0/source.json",
        "{",
        "  \"url\": \"https://example.com/archive.jar?with=query\",",
        "  \"integrity\": \"sha256-bleh\",",
        "  \"patches\": {",
        "    \"1.fix-this.patch\": \"sha256-lol\",",
        "    \"2.fix-that.patch\": \"sha256-kek\"",
        "  },",
        "  \"patch_strip\": 3",
        "}");
    server.start();

    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), RepositoryName.create("foorepo"), reporter))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
                .setRepoName("foorepo")
                .setUrls(
                    ImmutableList.of(
                        "https://mirror.bazel.build/mysite.com/thing.zip",
                        "file:///home/bazel/mymirror/mysite.com/thing.zip",
                        "http://mysite.com/thing.zip"))
                .setIntegrity("sha256-blah")
                .setStripPrefix("pref")
                .setRemotePatches(ImmutableMap.of())
                .setRemotePatchStrip(0)
                .build());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("bar", "2.0"), RepositoryName.create("barrepo"), reporter))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
                .setRepoName("barrepo")
                .setUrls(
                    ImmutableList.of(
                        "https://mirror.bazel.build/example.com/archive.jar?with=query",
                        "file:///home/bazel/mymirror/example.com/archive.jar?with=query",
                        "https://example.com/archive.jar?with=query"))
                .setIntegrity("sha256-bleh")
                .setStripPrefix("")
                .setRemotePatches(
                    ImmutableMap.<String, String>of(
                        server.getUrl() + "/modules/bar/2.0/patches/1.fix-this.patch", "sha256-lol",
                        server.getUrl() + "/modules/bar/2.0/patches/2.fix-that.patch",
                            "sha256-kek"))
                .setRemotePatchStrip(3)
                .build());
  }

  @Test
  public void testGetRepoInvalidRegistryJsonSpec() throws Exception {
    server.serve("/bazel_registry.json", "", "", "", "");
    server.start();
    server.serve(
        "/modules/foo/1.0/source.json",
        "{",
        "  \"url\": \"http://mysite.com/thing.zip\",",
        "  \"integrity\": \"sha256-blah\",",
        "  \"strip_prefix\": \"pref\"",
        "}");

    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), RepositoryName.create("foorepo"), reporter))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
                .setRepoName("foorepo")
                .setUrls(ImmutableList.of("http://mysite.com/thing.zip"))
                .setIntegrity("sha256-blah")
                .setStripPrefix("pref")
                .setRemotePatches(ImmutableMap.of())
                .setRemotePatchStrip(0)
                .build());
  }

  @Test
  public void testGetRepoInvalidModuleJsonSpec() throws Exception {
    server.serve(
        "/bazel_registry.json",
        "{",
        "  \"mirrors\": [",
        "    \"https://mirror.bazel.build/\",",
        "    \"file:///home/bazel/mymirror/\"",
        "  ]",
        "}");
    server.serve(
        "/modules/foo/1.0/source.json",
        "{",
        "  \"url\": \"http://mysite.com/thing.zip\",",
        "  \"integrity\": \"sha256-blah\",",
        "  \"strip_prefix\": \"pref\",",
        "}");
    server.start();

    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl());
    assertThrows(
        IOException.class,
        () ->
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), RepositoryName.create("foorepo"), reporter));
  }
}
