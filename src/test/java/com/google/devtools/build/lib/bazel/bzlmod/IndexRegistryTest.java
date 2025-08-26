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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.eventbus.Subscribe;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.authandtls.BasicHttpAuthenticationEncoder;
import com.google.devtools.build.lib.authandtls.Netrc;
import com.google.devtools.build.lib.authandtls.NetrcCredentials;
import com.google.devtools.build.lib.authandtls.NetrcParser;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
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
  private static class EventRecorder {
    private final List<RegistryFileDownloadEvent> downloadEvents = new ArrayList<>();

    @Subscribe
    public void onRegistryFileDownloadEvent(RegistryFileDownloadEvent downloadEvent) {
      downloadEvents.add(downloadEvent);
    }

    public ImmutableMap<String, Optional<Checksum>> getRecordedHashes() {
      return downloadEvents.stream()
          .collect(
              toImmutableMap(RegistryFileDownloadEvent::uri, RegistryFileDownloadEvent::checksum));
    }
  }

  private final String authToken = BasicHttpAuthenticationEncoder.encode("rinne", "rinnepass");
  private DownloadManager downloadManager;
  private EventRecorder eventRecorder;
  @Rule public final TestHttpServer server = new TestHttpServer(authToken);
  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

  private RegistryFactoryImpl registryFactory;
  private DownloadCache downloadCache;

  @Before
  public void setUp() throws Exception {
    eventRecorder = new EventRecorder();
    eventBus.register(eventRecorder);
    downloadCache = new DownloadCache();
    HttpDownloader httpDownloader = new HttpDownloader();
    downloadManager = new DownloadManager(downloadCache, httpDownloader, httpDownloader);
    registryFactory = new RegistryFactoryImpl(Suppliers.ofInstance(ImmutableMap.of()));
  }

  @Test
  public void testHttpUrl() throws Exception {
    server.serve("/myreg/modules/foo/1.0/MODULE.bazel", "lol");
    server.start();

    Registry registry =
        registryFactory.createRegistry(
            server.getUrl() + "/myreg",
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    assertThat(registry.getModuleFile(createModuleKey("foo", "1.0"), reporter, downloadManager))
        .isEqualTo(
            ModuleFile.create(
                "lol".getBytes(UTF_8), server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel"));
    assertThrows(
        Registry.NotFoundException.class,
        () -> registry.getModuleFile(createModuleKey("bar", "1.0"), reporter, downloadManager));
  }

  @Test
  public void testHttpUrlWithNetrcCreds() throws Exception {
    server.serve("/myreg/modules/foo/1.0/MODULE.bazel", "lol".getBytes(UTF_8), true);
    server.start();
    Netrc netrc =
        NetrcParser.parseAndClose(
            new ByteArrayInputStream(
                "machine [::1] login rinne password rinnepass\n".getBytes(UTF_8)));
    Registry registry =
        registryFactory.createRegistry(
            server.getUrl() + "/myreg",
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());

    var e =
        assertThrows(
            IOException.class,
            () -> registry.getModuleFile(createModuleKey("foo", "1.0"), reporter, downloadManager));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Failed to fetch registry file %s: GET returned 401 Unauthorized"
                .formatted(server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel"));

    downloadManager.setNetrcCreds(new NetrcCredentials(netrc));
    assertThat(registry.getModuleFile(createModuleKey("foo", "1.0"), reporter, downloadManager))
        .isEqualTo(
            ModuleFile.create(
                "lol".getBytes(UTF_8), server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel"));
    assertThrows(
        Registry.NotFoundException.class,
        () -> registry.getModuleFile(createModuleKey("bar", "1.0"), reporter, downloadManager));
  }

  @Test
  public void testFileUrl() throws Exception {
    tempFolder.newFolder("fakereg", "modules", "foo", "1.0");
    File file = tempFolder.newFile("fakereg/modules/foo/1.0/MODULE.bazel");
    try (Writer writer = Files.newBufferedWriter(file.toPath(), UTF_8)) {
      writer.write("lol");
    }

    Registry registry =
        registryFactory.createRegistry(
            new File(tempFolder.getRoot(), "fakereg").toURI().toString(),
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    assertThat(registry.getModuleFile(createModuleKey("foo", "1.0"), reporter, downloadManager))
        .isEqualTo(ModuleFile.create("lol".getBytes(UTF_8), file.toURI().toString()));
    assertThrows(
        Registry.NotFoundException.class,
        () -> registry.getModuleFile(createModuleKey("bar", "1.0"), reporter, downloadManager));
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
        "  \"mirror_urls\":"
            + " [\"http://my.mirror/mysite.com/thing.zip\",\"http://another.mirror/mysite.com/thing.zip\"],",
        "  \"integrity\": \"sha256-blah\",",
        "  \"strip_prefix\": \"pref\"",
        "}");
    server.serve("/modules/foo/1.0/MODULE.bazel", "module(name = \"foo\", version = \"1.0\")");
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
    server.serve("/modules/bar/2.0/MODULE.bazel", "module(name = \"bar\", version = \"2.0\")");
    server.serve(
        "/modules/baz/3.0/source.json",
        """
        {
            "url": "https://example.com/archive.jar?with=query",
            "integrity": "sha256-bleh",
            "overlay": {
                "BUILD.bazel": "sha256-bleh-overlay"
            }
        }
        """);
    server.serve("/modules/baz/3.0/MODULE.bazel", "module(name = \"baz\", version = \"3.0\")");
    server.start();
    var moduleFileRegistryHashes =
        ImmutableMap.of(
            server.getUrl() + "/modules/foo/1.0/MODULE.bazel",
            Optional.of(sha256("module(name = \"foo\", version = \"1.0\")")),
            server.getUrl() + "/modules/baz/3.0/MODULE.bazel",
            Optional.of(sha256("module(name = \"baz\", version = \"3.0\")")),
            server.getUrl() + "/modules/bar/2.0/MODULE.bazel",
            Optional.of(sha256("module(name = \"bar\", version = \"2.0\")")));

    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(),
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), moduleFileRegistryHashes, reporter, downloadManager))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
                .setUrls(
                    ImmutableList.of(
                        "https://mirror.bazel.build/mysite.com/thing.zip",
                        "file:///home/bazel/mymirror/mysite.com/thing.zip",
                        "http://mysite.com/thing.zip",
                        "http://my.mirror/mysite.com/thing.zip",
                        "http://another.mirror/mysite.com/thing.zip"))
                .setIntegrity("sha256-blah")
                .setStripPrefix("pref")
                .setRemotePatches(ImmutableMap.of())
                .setOverlay(ImmutableMap.of())
                .setRemoteModuleFile(
                    new ArchiveRepoSpecBuilder.RemoteFile(
                        sha256("module(name = \"foo\", version = \"1.0\")")
                            .toSubresourceIntegrity(),
                        ImmutableList.of(server.getUrl() + "/modules/foo/1.0/MODULE.bazel")))
                .setRemotePatchStrip(0)
                .build());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("bar", "2.0"), moduleFileRegistryHashes, reporter, downloadManager))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
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
                .setOverlay(ImmutableMap.of())
                .setRemoteModuleFile(
                    new ArchiveRepoSpecBuilder.RemoteFile(
                        sha256("module(name = \"bar\", version = \"2.0\")")
                            .toSubresourceIntegrity(),
                        ImmutableList.of(server.getUrl() + "/modules/bar/2.0/MODULE.bazel")))
                .build());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("baz", "3.0"), moduleFileRegistryHashes, reporter, downloadManager))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
                .setUrls(
                    ImmutableList.of(
                        "https://mirror.bazel.build/example.com/archive.jar?with=query",
                        "file:///home/bazel/mymirror/example.com/archive.jar?with=query",
                        "https://example.com/archive.jar?with=query"))
                .setIntegrity("sha256-bleh")
                .setStripPrefix("")
                .setOverlay(
                    ImmutableMap.of(
                        "BUILD.bazel",
                        new ArchiveRepoSpecBuilder.RemoteFile(
                            "sha256-bleh-overlay",
                            // URLs in the registry itself are not mirrored.
                            ImmutableList.of(
                                server.getUrl() + "/modules/baz/3.0/overlay/BUILD.bazel"))))
                .setRemoteModuleFile(
                    new ArchiveRepoSpecBuilder.RemoteFile(
                        sha256("module(name = \"baz\", version = \"3.0\")")
                            .toSubresourceIntegrity(),
                        ImmutableList.of(server.getUrl() + "/modules/baz/3.0/MODULE.bazel")))
                .setRemotePatches(ImmutableMap.of())
                .setRemotePatchStrip(0)
                .build());
  }

  @Test
  public void testGetGitRepoSpec() throws Exception {
    server.serve(
        "/bazel_registry.json",
        """
        {
          "mirrors": [
            "https://mirror.bazel.build/",
            "file:///home/bazel/mymirror/"
          ]
        }
        """);
    server.serve(
        "/modules/foo/1.0/source.json",
        """
        {
            "type": "git_repository",
            "remote": "https://github.com/raspberrypi/pico-sdk.git",
            "commit": "4b6e647590213f253f2789ad9026df1d00f38c5d"
        }
        """);
    server.serve("/modules/foo/1.0/MODULE.bazel", "module(name = \"foo\", version = \"1.0\")");
    server.start();
    var moduleFileRegistryHashes =
        ImmutableMap.of(
            server.getUrl() + "/modules/foo/1.0/MODULE.bazel",
            Optional.of(sha256("module(name = \"foo\", version = \"1.0\")")));

    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(),
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), moduleFileRegistryHashes, reporter, downloadManager))
        .isEqualTo(
            new GitRepoSpecBuilder()
                .setRemote("https://github.com/raspberrypi/pico-sdk.git")
                .setCommit("4b6e647590213f253f2789ad9026df1d00f38c5d")
                .setInitSubmodules(false)
                .setVerbose(false)
                .setRemoteModuleFile(
                    new ArchiveRepoSpecBuilder.RemoteFile(
                        sha256("module(name = \"foo\", version = \"1.0\")")
                            .toSubresourceIntegrity(),
                        ImmutableList.of(server.getUrl() + "/modules/foo/1.0/MODULE.bazel")))
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

    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(),
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), ImmutableMap.of(), reporter, downloadManager))
        .isEqualTo(LocalPathRepoSpecs.create("/hello/bar/project_x"));
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
    server.serve("/modules/foo/1.0/MODULE.bazel", "module(name = \"foo\", version = \"1.0\")");

    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(),
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"),
                ImmutableMap.of(
                    server.getUrl() + "/modules/foo/1.0/MODULE.bazel",
                    Optional.of(sha256("module(name = \"foo\", version = \"1.0\")"))),
                reporter,
                downloadManager))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
                .setUrls(ImmutableList.of("http://mysite.com/thing.zip"))
                .setIntegrity("sha256-blah")
                .setStripPrefix("pref")
                .setRemotePatches(ImmutableMap.of())
                .setOverlay(ImmutableMap.of())
                .setRemoteModuleFile(
                    new ArchiveRepoSpecBuilder.RemoteFile(
                        sha256("module(name = \"foo\", version = \"1.0\")")
                            .toSubresourceIntegrity(),
                        ImmutableList.of(server.getUrl() + "/modules/foo/1.0/MODULE.bazel")))
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

    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(),
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    assertThrows(
        IOException.class,
        () ->
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), ImmutableMap.of(), reporter, downloadManager));
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
    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(),
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    Optional<ImmutableMap<Version, String>> yankedVersion =
        registry.getYankedVersions("red-pill", reporter, downloadManager);
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
    server.serve(
        "/modules/archive_type/1.0/MODULE.bazel",
        "module(name = \"archive_type\", version = \"1.0\")");
    server.start();

    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(),
            LockfileMode.UPDATE,
            ImmutableMap.of(),
            ImmutableMap.of(),
            Optional.empty());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("archive_type", "1.0"),
                ImmutableMap.of(
                    server.getUrl() + "/modules/archive_type/1.0/MODULE.bazel",
                    Optional.of(sha256("module(name = \"archive_type\", version = \"1.0\")"))),
                reporter,
                downloadManager))
        .isEqualTo(
            new ArchiveRepoSpecBuilder()
                .setUrls(ImmutableList.of("https://mysite.com/thing?format=zip"))
                .setIntegrity("sha256-blah")
                .setStripPrefix("")
                .setArchiveType("zip")
                .setRemotePatches(ImmutableMap.of())
                .setRemotePatchStrip(0)
                .setOverlay(ImmutableMap.of())
                .setRemoteModuleFile(
                    new ArchiveRepoSpecBuilder.RemoteFile(
                        sha256("module(name = \"archive_type\", version = \"1.0\")")
                            .toSubresourceIntegrity(),
                        ImmutableList.of(
                            server.getUrl() + "/modules/archive_type/1.0/MODULE.bazel")))
                .build());
  }

  @Test
  public void testGetModuleFileChecksums() throws Exception {
    downloadCache.setPath(scratch.dir("cache"));

    server.serve("/myreg/modules/foo/1.0/MODULE.bazel", "old");
    server.serve("/myreg/modules/foo/2.0/MODULE.bazel", "new");
    server.start();

    var knownFiles =
        ImmutableMap.of(
            server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel",
            Optional.of(sha256("old")),
            server.getUrl() + "/myreg/modules/unused/1.0/MODULE.bazel",
            Optional.of(sha256("unused")));
    Registry registry =
        registryFactory.createRegistry(
            server.getUrl() + "/myreg",
            LockfileMode.UPDATE,
            knownFiles,
            ImmutableMap.of(),
            Optional.empty());
    assertThat(registry.getModuleFile(createModuleKey("foo", "1.0"), reporter, downloadManager))
        .isEqualTo(
            ModuleFile.create(
                "old".getBytes(UTF_8), server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel"));
    assertThat(registry.getModuleFile(createModuleKey("foo", "2.0"), reporter, downloadManager))
        .isEqualTo(
            ModuleFile.create(
                "new".getBytes(UTF_8), server.getUrl() + "/myreg/modules/foo/2.0/MODULE.bazel"));
    var e =
        assertThrows(
            Registry.NotFoundException.class,
            () -> registry.getModuleFile(createModuleKey("bar", "1.0"), reporter, downloadManager));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(server.getUrl() + "/myreg/modules/bar/1.0/MODULE.bazel: not found");

    var recordedChecksums = eventRecorder.getRecordedHashes();
    assertThat(
            Maps.transformValues(
                recordedChecksums, maybeChecksum -> maybeChecksum.map(Checksum::toString)))
        .containsExactly(
            server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel",
            Optional.of(sha256("old").toString()),
            server.getUrl() + "/myreg/modules/foo/2.0/MODULE.bazel",
            Optional.of(sha256("new").toString()),
            server.getUrl() + "/myreg/modules/bar/1.0/MODULE.bazel",
            Optional.empty())
        .inOrder();

    Registry registry2 =
        registryFactory.createRegistry(
            server.getUrl() + "/myreg",
            LockfileMode.UPDATE,
            recordedChecksums,
            ImmutableMap.of(),
            Optional.empty());
    // Test that the recorded hashes are used for repo cache hits even when the server content
    // changes.
    server.unserve("/myreg/modules/foo/1.0/MODULE.bazel");
    server.unserve("/myreg/modules/foo/2.0/MODULE.bazel");
    server.serve("/myreg/modules/bar/1.0/MODULE.bazel", "no longer 404");
    assertThat(registry2.getModuleFile(createModuleKey("foo", "1.0"), reporter, downloadManager))
        .isEqualTo(
            ModuleFile.create(
                "old".getBytes(UTF_8), server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel"));
    assertThat(registry2.getModuleFile(createModuleKey("foo", "2.0"), reporter, downloadManager))
        .isEqualTo(
            ModuleFile.create(
                "new".getBytes(UTF_8), server.getUrl() + "/myreg/modules/foo/2.0/MODULE.bazel"));
    e =
        assertThrows(
            Registry.NotFoundException.class,
            () ->
                registry2.getModuleFile(createModuleKey("bar", "1.0"), reporter, downloadManager));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            server.getUrl()
                + "/myreg/modules/bar/1.0/MODULE.bazel: previously not found (as recorded in"
                + " MODULE.bazel.lock, refresh with --lockfile_mode=refresh)");
  }

  @Test
  public void testGetModuleFileChecksumMismatch() throws Exception {
    downloadCache.setPath(scratch.dir("cache"));

    server.serve("/myreg/modules/foo/1.0/MODULE.bazel", "fake");
    server.start();

    var knownFiles =
        ImmutableMap.of(
            server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel",
            Optional.of(sha256("original")));
    Registry registry =
        registryFactory.createRegistry(
            server.getUrl() + "/myreg",
            LockfileMode.UPDATE,
            knownFiles,
            ImmutableMap.of(),
            Optional.empty());
    var e =
        assertThrows(
            IOException.class,
            () -> registry.getModuleFile(createModuleKey("foo", "1.0"), reporter, downloadManager));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Failed to fetch registry file %s: Checksum was %s but wanted %s"
                .formatted(
                    server.getUrl() + "/myreg/modules/foo/1.0/MODULE.bazel",
                    sha256("fake"),
                    sha256("original")));
  }

  @Test
  public void testGetRepoSpecChecksum() throws Exception {
    downloadCache.setPath(scratch.dir("cache"));

    String registryJson =
        """
        {
          "module_base_path": "/hello/foo"
        }
        """;
    server.serve("/bazel_registry.json", registryJson);
    String sourceJson =
        """
        {
          "type": "local_path",
          "path": "../bar/project_x"
        }
        """;
    server.serve("/modules/foo/1.0/source.json", sourceJson.getBytes(UTF_8));
    server.start();

    var knownFiles =
        ImmutableMap.of(
            server.getUrl() + "/modules/foo/2.0/source.json", Optional.of(sha256("unused")));
    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(), LockfileMode.UPDATE, knownFiles, ImmutableMap.of(), Optional.empty());
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), ImmutableMap.of(), reporter, downloadManager))
        .isEqualTo(LocalPathRepoSpecs.create("/hello/bar/project_x"));

    var recordedChecksums = eventRecorder.getRecordedHashes();
    assertThat(
            Maps.transformValues(recordedChecksums, checksum -> checksum.map(Checksum::toString)))
        .containsExactly(
            server.getUrl() + "/bazel_registry.json",
            Optional.of(sha256(registryJson).toString()),
            server.getUrl() + "/modules/foo/1.0/source.json",
            Optional.of(sha256(sourceJson).toString()));

    registry =
        registryFactory.createRegistry(
            server.getUrl(),
            LockfileMode.UPDATE,
            recordedChecksums,
            ImmutableMap.of(),
            Optional.empty());
    // Test that the recorded hashes are used for repo cache hits even when the server content
    // changes.
    server.unserve("/bazel_registry.json");
    server.unserve("/modules/foo/1.0/source.json");
    assertThat(
            registry.getRepoSpec(
                createModuleKey("foo", "1.0"), ImmutableMap.of(), reporter, downloadManager))
        .isEqualTo(LocalPathRepoSpecs.create("/hello/bar/project_x"));
  }

  @Test
  public void testGetRepoSpecChecksumMismatch() throws Exception {
    downloadCache.setPath(scratch.dir("cache"));

    String registryJson =
        """
        {
          "module_base_path": "/hello/foo"
        }
        """;
    server.serve("/bazel_registry.json", registryJson.getBytes(UTF_8));
    String sourceJson =
        """
        {
          "type": "local_path",
          "path": "../bar/project_x"
        }
        """;
    String maliciousSourceJson = sourceJson.replace("project_x", "malicious");
    server.serve("/modules/foo/1.0/source.json", maliciousSourceJson.getBytes(UTF_8));
    server.start();

    var knownFiles =
        ImmutableMap.of(
            server.getUrl() + "/bazel_registry.json",
            Optional.of(sha256(registryJson)),
            server.getUrl() + "/modules/foo/1.0/source.json",
            Optional.of(sha256(sourceJson)));
    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(), LockfileMode.UPDATE, knownFiles, ImmutableMap.of(), Optional.empty());
    var e =
        assertThrows(
            IOException.class,
            () ->
                registry.getRepoSpec(
                    createModuleKey("foo", "1.0"), ImmutableMap.of(), reporter, downloadManager));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Failed to fetch registry file %s: Checksum was %s but wanted %s"
                .formatted(
                    server.getUrl() + "/modules/foo/1.0/source.json",
                    sha256(maliciousSourceJson),
                    sha256(sourceJson)));
  }

  @Test
  public void testBazelRegistryChecksumMismatch() throws Exception {
    downloadCache.setPath(scratch.dir("cache"));

    String registryJson =
        """
        {
          "module_base_path": "/hello/foo"
        }
        """;
    String maliciousRegistryJson = registryJson.replace("foo", "malicious");
    server.serve("/bazel_registry.json", maliciousRegistryJson.getBytes(UTF_8));
    String sourceJson =
        """
        {
          "type": "local_path",
          "path": "../bar/project_x"
        }
        """;
    server.serve("/modules/foo/1.0/source.json", sourceJson.getBytes(UTF_8));
    server.start();

    var knownFiles =
        ImmutableMap.of(
            server.getUrl() + "/bazel_registry.json",
            Optional.of(sha256(registryJson)),
            server.getUrl() + "/modules/foo/1.0/source.json",
            Optional.of(sha256(sourceJson)));
    Registry registry =
        registryFactory.createRegistry(
            server.getUrl(), LockfileMode.UPDATE, knownFiles, ImmutableMap.of(), Optional.empty());
    var e =
        assertThrows(
            IOException.class,
            () ->
                registry.getRepoSpec(
                    createModuleKey("foo", "1.0"), ImmutableMap.of(), reporter, downloadManager));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Failed to fetch registry file %s: Checksum was %s but wanted %s"
                .formatted(
                    server.getUrl() + "/bazel_registry.json",
                    sha256(maliciousRegistryJson),
                    sha256(registryJson)));
  }

  private static Checksum sha256(String content) throws Checksum.InvalidChecksumException {
    return Checksum.fromString(
        DownloadCache.KeyType.SHA256, Hashing.sha256().hashString(content, UTF_8).toString());
  }
}
