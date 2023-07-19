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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.gson.FieldNamingPolicy;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;

/**
 * Represents a Bazel module registry that serves a list of module metadata from a static HTTP
 * server or a local file path.
 *
 * <p>For details, see <a href="https://bazel.build/external/registry">the docs</a>
 */
public class IndexRegistry implements Registry {
  private final URI uri;
  private final DownloadManager downloadManager;
  private final Map<String, String> clientEnv;
  private final Gson gson;

  public IndexRegistry(URI uri, DownloadManager downloadManager, Map<String, String> clientEnv) {
    this.uri = uri;
    this.downloadManager = downloadManager;
    this.clientEnv = clientEnv;
    this.gson =
        new GsonBuilder()
            .setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
            .create();
  }

  @Override
  public String getUrl() {
    return uri.toString();
  }

  private String constructUrl(String base, String... segments) {
    StringBuilder url = new StringBuilder(base);
    for (String segment : segments) {
      if (url.charAt(url.length() - 1) != '/' && !segment.startsWith("/")) {
        url.append('/');
      }
      url.append(segment);
    }
    return url.toString();
  }

  /** Grabs a file from the given URL. Returns {@link Optional#empty} if the file doesn't exist. */
  private Optional<byte[]> grabFile(String url, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    try {
      return Optional.of(
          downloadManager.downloadAndReadOneUrl(new URL(url), eventHandler, clientEnv));
    } catch (FileNotFoundException e) {
      return Optional.empty();
    }
  }

  @Override
  public Optional<ModuleFile> getModuleFile(ModuleKey key, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    String url =
        constructUrl(
            getUrl(), "modules", key.getName(), key.getVersion().toString(), "MODULE.bazel");
    return grabFile(url, eventHandler).map(content -> ModuleFile.create(content, url));
  }

  /** Represents fields available in {@code bazel_registry.json} for the registry. */
  private static class BazelRegistryJson {
    String[] mirrors;
    String moduleBasePath;
  }

  /** Represents fields available in {@code source.json} for each version of a module. */
  private static class SourceJson {
    String type = "archive";
    URL url;
    String integrity;
    String stripPrefix;
    Map<String, String> patches;
    int patchStrip;
    String path;
    String archiveType;
  }

  /**
   * Grabs a JSON file from the given URL, and returns it as a parsed object with fields in {@code
   * T}. Returns {@link Optional#empty} if the file doesn't exist.
   */
  private <T> Optional<T> grabJson(String url, Class<T> klass, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    Optional<byte[]> bytes = grabFile(url, eventHandler);
    if (bytes.isEmpty()) {
      return Optional.empty();
    }
    String jsonString = new String(bytes.get(), UTF_8);
    if (jsonString.isBlank()) {
      return Optional.empty();
    }
    try {
      return Optional.of(gson.fromJson(jsonString, klass));
    } catch (JsonParseException e) {
      throw new IOException(
          String.format("Unable to parse json at url %s: %s", url, e.getMessage()), e);
    }
  }

  @Override
  public RepoSpec getRepoSpec(
      ModuleKey key, RepositoryName repoName, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    Optional<BazelRegistryJson> bazelRegistryJson =
        grabJson(
            constructUrl(getUrl(), "bazel_registry.json"), BazelRegistryJson.class, eventHandler);
    Optional<SourceJson> sourceJson =
        grabJson(
            constructUrl(
                getUrl(), "modules", key.getName(), key.getVersion().toString(), "source.json"),
            SourceJson.class,
            eventHandler);
    if (sourceJson.isEmpty()) {
      throw new FileNotFoundException(
          String.format("Module %s's source information not found in registry %s", key, getUrl()));
    }

    String type = sourceJson.get().type;
    switch (type) {
      case "archive":
        return createArchiveRepoSpec(sourceJson, bazelRegistryJson, key, repoName);
      case "local_path":
        return createLocalPathRepoSpec(sourceJson, bazelRegistryJson, key, repoName);
      default:
        throw new IOException(String.format("Invalid source type for module %s", key));
    }
  }

  private RepoSpec createLocalPathRepoSpec(
      Optional<SourceJson> sourceJson,
      Optional<BazelRegistryJson> bazelRegistryJson,
      ModuleKey key,
      RepositoryName repoName)
      throws IOException {
    String path = sourceJson.get().path;
    if (!PathFragment.isAbsolute(path)) {
      String moduleBase = bazelRegistryJson.get().moduleBasePath;
      path = moduleBase + "/" + path;
      if (!PathFragment.isAbsolute(moduleBase)) {
        if (uri.getScheme().equals("file")) {
          if (uri.getPath().isEmpty() || !uri.getPath().startsWith("/")) {
            throw new IOException(
                String.format(
                    "Provided non absolute local registry path for module %s: %s",
                    key, uri.getPath()));
          }
          // Unix:    file:///tmp --> /tmp
          // Windows: file:///C:/tmp --> C:/tmp
          path = uri.getPath().substring(OS.getCurrent() == OS.WINDOWS ? 1 : 0) + "/" + path;
        } else {
          throw new IOException(String.format("Provided non local registry for module %s", key));
        }
      }
    }

    return RepoSpec.builder()
        .setRuleClassName("local_repository")
        .setAttributes(
            AttributeValues.create(
                ImmutableMap.of(
                    "name", repoName.getName(), "path", PathFragment.create(path).toString())))
        .build();
  }

  private RepoSpec createArchiveRepoSpec(
      Optional<SourceJson> sourceJson,
      Optional<BazelRegistryJson> bazelRegistryJson,
      ModuleKey key,
      RepositoryName repoName)
      throws IOException {
    URL sourceUrl = sourceJson.get().url;
    if (sourceUrl == null) {
      throw new IOException(String.format("Missing source URL for module %s", key));
    }
    if (sourceJson.get().integrity == null) {
      throw new IOException(String.format("Missing integrity for module %s", key));
    }

    ImmutableList.Builder<String> urls = new ImmutableList.Builder<>();
    // For each mirror specified in bazel_registry.json, add a URL that's essentially the mirror
    // URL concatenated with the source URL.
    if (bazelRegistryJson.isPresent() && bazelRegistryJson.get().mirrors != null) {
      for (String mirror : bazelRegistryJson.get().mirrors) {
        try {
          var unused = new URL(mirror);
        } catch (MalformedURLException e) {
          throw new IOException("Malformed mirror URL", e);
        }

        urls.add(constructUrl(mirror, sourceUrl.getAuthority(), sourceUrl.getFile()));
      }
    }
    // Finally add the original source URL itself.
    urls.add(sourceUrl.toString());

    // Build remote patches as key-value pairs of "url" => "integrity".
    ImmutableMap.Builder<String, String> remotePatches = new ImmutableMap.Builder<>();
    if (sourceJson.get().patches != null) {
      for (Map.Entry<String, String> entry : sourceJson.get().patches.entrySet()) {
        remotePatches.put(
            constructUrl(
                getUrl(),
                "modules",
                key.getName(),
                key.getVersion().toString(),
                "patches",
                entry.getKey()),
            entry.getValue());
      }
    }

    return new ArchiveRepoSpecBuilder()
        .setRepoName(repoName.getName())
        .setUrls(urls.build())
        .setIntegrity(sourceJson.get().integrity)
        .setStripPrefix(Strings.nullToEmpty(sourceJson.get().stripPrefix))
        .setRemotePatches(remotePatches.buildOrThrow())
        .setRemotePatchStrip(sourceJson.get().patchStrip)
        .setArchiveType(sourceJson.get().archiveType)
        .build();
  }

  @Override
  public Optional<ImmutableMap<Version, String>> getYankedVersions(
      String moduleName, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    Optional<MetadataJson> metadataJson =
        grabJson(
            constructUrl(getUrl(), "modules", moduleName, "metadata.json"),
            MetadataJson.class,
            eventHandler);
    if (metadataJson.isEmpty()) {
      return Optional.empty();
    }

    try {
      ImmutableMap.Builder<Version, String> yankedVersionsBuilder = new ImmutableMap.Builder<>();
      if (metadataJson.get().yankedVersions != null) {
        for (Entry<String, String> e : metadataJson.get().yankedVersions.entrySet()) {
          yankedVersionsBuilder.put(Version.parse(e.getKey()), e.getValue());
        }
      }
      return Optional.of(yankedVersionsBuilder.buildOrThrow());
    } catch (ParseException e) {
      throw new IOException(
          String.format(
              "Could not parse module %s's metadata file: %s", moduleName, e.getMessage()));
    }
  }

  /** Represents fields available in {@code metadata.json} for each module. */
  static class MetadataJson {
    // There are other attributes in the metadata.json file, but for now, we only care about
    // the yanked_version attribute.
    Map<String, String> yankedVersions;
  }
}
