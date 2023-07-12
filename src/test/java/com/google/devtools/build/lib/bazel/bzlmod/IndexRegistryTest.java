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
import com.google.devtools.build.lib.authandtls.BasicHttpAuthenticationEncoder;
import com.google.devtools.build.lib.authandtls.Netrc;
import com.google.devtools.build.lib.authandtls.NetrcCredentials;
import com.google.devtools.build.lib.authandtls.NetrcParser;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.repository.downloader.UnrecoverableHttpException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.util.Optional;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link IndexRegistry}. */
@RunWith(JUnit4.class)
public class IndexRegistryTest extends FoundationTestCase {
  private final String authToken =
      BasicHttpAuthenticationEncoder.encode("rinne", "rinnepass", UTF_8);
  private DownloadManager downloadManager;
  @Rule public final TestHttpServer server = new TestHttpServer(authToken);
  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

  private RegistryFactory registryFactory;

  @Before
  public void setUp() throws Exception {
    downloadManager = new DownloadManager(new RepositoryCache(), new HttpDownloader());
    registryFactory =
        new RegistryFactoryImpl(downloadManager, Suppliers.ofInstance(ImmutableMap.of()));
  }

  @Test
  public void testHttpUrl() throws Exception {
    server.serve("/myreg/modules/foo/1.0/MODULE.bazel", "lol");
    server.start();

    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl() + "/myreg");
    assertThat(registry.getModuleFile(createModuleKey("foo", "1.0"), reporter))
        .hasValue(
            ModuleFile.create(
                "lol".getBytes(UTF_8), server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel"));
    assertThat(registry.getModuleFile(createModuleKey("bar", "1.0"), reporter)).isEmpty();
  }

  @Test
  public void testHttpUrlWithNetrcCreds() throws Exception {
    server.serve("/myreg/modules/foo/1.0/MODULE.bazel", "lol".getBytes(UTF_8), true);
    server.start();
    Netrc netrc =
        NetrcParser.parseAndClose(
            new ByteArrayInputStream(
                "machine [::1] login rinne password rinnepass\n".getBytes(UTF_8)));
    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl() + "/myreg");

    UnrecoverableHttpException e =
        assertThrows(
            UnrecoverableHttpException.class,
            () -> registry.getModuleFile(createModuleKey("foo", "1.0"), reporter));
    assertThat(e).hasMessageThat().contains("GET returned 401 Unauthorized");

    downloadManager.setNetrcCreds(new NetrcCredentials(netrc));
    assertThat(registry.getModuleFile(createModuleKey("foo", "1.0"), reporter))
        .hasValue(
            ModuleFile.create(
                "lol".getBytes(UTF_8), server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel"));
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
        .hasValue(ModuleFile.create("lol".getBytes(UTF_8), file.toURI().toString()));
    assertThat(registry.getModuleFile(createModuleKey("bar", "1.0"), reporter)).isEmpty();
  }

  @Test
  public void testGetArchiveRepoSpec() throws Exception {
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
                    ImmutableMap.of(
                        server.getUrl() + "/modules/bar/2.0/patches/1.fix-this.patch", "sha256-lol",
                        server.getUrl() + "/modules/bar/2.0/patches/2.fix-that.patch",
                            "sha256-kek"))
                .setRemotePatchStrip(3)
                .build());
  }

  @Test
  public void testGetLocalPathRepoSpec() throws Exception {
    server.serve("/bazel_registry.json", "{", "  \"module_base_path\": \"/hello/foo\"", "}");
    server.serve(
        "/modules/foo/1.0/source.json",
        "{",
        "  \"type\": \"local_path\",",
        "  \"path\": \"../bar/project_x\"",
        "}");
    server.start();

    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), RepositoryName.create("foorepo"), reporter))
        .isEqualTo(
            RepoSpec.builder()
                .setRuleClassName("local_repository")
                .setAttributes(
                    AttributeValues.create(
                        ImmutableMap.of("name", "foorepo", "path", "/hello/bar/project_x")))
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

  @Test
  public void testGetYankedVersion() throws Exception {
    server.serve(
        "/modules/red-pill/metadata.json",
        "{\n"
            + "    'homepage': 'https://docs.matrix.org/red-pill',\n"
            + "    'maintainers': [\n"
            + "        {\n"
            + "            'email': 'neo@matrix.org',\n"
            + "            'github': 'neo',\n"
            + "            'name': 'Neo'\n"
            + "        }\n"
            + "    ],\n"
            + "    'versions': [\n"
            + "        '1.0',\n"
            + "        '2.0'\n"
            + "    ],\n"
            + "    'yanked_versions': {"
            + "        '1.0': 'red-pill 1.0 is yanked due to CVE-2000-101, please upgrade to 2.0'\n"
            + "    }\n"
            + "}");
    server.start();
    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl());
    Optional<ImmutableMap<Version, String>> yankedVersion =
        registry.getYankedVersions("red-pill", reporter);
    assertThat(yankedVersion)
        .hasValue(
            ImmutableMap.of(
                Version.parse("1.0"),
                "red-pill 1.0 is yanked due to CVE-2000-101, please upgrade to 2.0"));
  }

  @Test
  public void testArchiveWithExplicitType() throws Exception {
    server.serve(
        "/modules/archive_type/1.0/source.json",
        "{",
        "  \"url\": \"https://mysite.com/thing?format=zip\",",
        "  \"integrity\": \"sha256-blah\",",
        "  \"archive_type\": \"zip\"",
        "}");
    server.start();

    Registry registry = registryFactory.getRegistryWithUrl(server.getUrl());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("archive_type", "1.0"),
                RepositoryName.create("archive_type_repo"),
                reporter))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
                .setRepoName("archive_type_repo")
                .setUrls(ImmutableList.of("https://mysite.com/thing?format=zip"))
                .setIntegrity("sha256-blah")
                .setStripPrefix("")
                .setArchiveType("zip")
                .setRemotePatches(ImmutableMap.of())
                .setRemotePatchStrip(0)
                .build());
  }
}
