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
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.gson.FieldNamingPolicy;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.util.Map;
import java.util.Optional;

/**
 * Represents a Bazel module registry that serves a list of module metadata from a static HTTP
 * server or a local file path.
 */
// TODO(wyv): Insert "For details, see ..." when we have public documentation.
public class IndexRegistry implements Registry {
  private final URI uri;
  private final HttpDownloader httpDownloader;
  private final Map<String, String> clientEnv;
  private final Gson gson;

  public IndexRegistry(URI uri, HttpDownloader httpDownloader, Map<String, String> clientEnv) {
    this.uri = uri;
    this.httpDownloader = httpDownloader;
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
          httpDownloader.downloadAndReadOneUrl(new URL(url), eventHandler, clientEnv));
    } catch (FileNotFoundException e) {
      return Optional.empty();
    }
  }

  @Override
  public Optional<byte[]> getModuleFile(ModuleKey key, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    return grabFile(
        constructUrl(
            getUrl(), "modules", key.getName(), key.getVersion().toString(), "MODULE.bazel"),
        eventHandler);
  }

  /** Represents fields available in {@code bazel_registry.json} for the registry. */
  private static class BazelRegistryJson {
    String[] mirrors;
  }

  /** Represents fields available in {@code source.json} for each version of a module. */
  private static class SourceJson {
    URL url;
    String integrity;
    String stripPrefix;
    Map<String, String> patches;
    int patchStrip;
  }

  /**
   * Grabs a JSON file from the given URL, and returns it as a parsed object with fields in {@code
   * T}. Returns {@link Optional#empty} if the file doesn't exist.
   */
  private <T> Optional<T> grabJson(String url, Class<T> klass, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    Optional<byte[]> bytes = grabFile(url, eventHandler);
    if (!bytes.isPresent()) {
      return Optional.empty();
    }
    String jsonString = new String(bytes.get(), UTF_8);
    return Optional.of(gson.fromJson(jsonString, klass));
  }

  @Override
  public RepoSpec getRepoSpec(ModuleKey key, String repoName, ExtendedEventHandler eventHandler)
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
    if (!sourceJson.isPresent()) {
      throw new FileNotFoundException(
          String.format("Module %s's source information not found in registry %s", key, getUrl()));
    }
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
          new URL(mirror);
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
        .setRepoName(repoName)
        .setUrls(urls.build())
        .setIntegrity(sourceJson.get().integrity)
        .setStripPrefix(Strings.nullToEmpty(sourceJson.get().stripPrefix))
        .setRemotePatches(remotePatches.buildOrThrow())
        .setRemotePatchStrip(sourceJson.get().patchStrip)
        .build();
  }
}
