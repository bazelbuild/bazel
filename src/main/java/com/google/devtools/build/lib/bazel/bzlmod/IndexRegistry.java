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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
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

  /**
   * How to handle the list of file hashes known from the lockfile when downloading files from the
   * registry.
   */
  public enum KnownFileHashesMode {
    IGNORE,
    USE_AND_UPDATE,
    ENFORCE
  }

  /** The unresolved version of the url. Ex: has %workspace% placeholder */
  private final String unresolvedUri;

  private final URI uri;
  private final DownloadManager downloadManager;
  private final Map<String, String> clientEnv;
  private final Gson gson;
  private final ImmutableMap<String, Optional<Checksum>> knownFileHashes;
  private final KnownFileHashesMode knownFileHashesMode;
  private volatile Optional<BazelRegistryJson> bazelRegistryJson;
  private volatile StoredEventHandler bazelRegistryJsonEvents;

  private static final String SOURCE_JSON_FILENAME = "source.json";

  public IndexRegistry(
      URI uri,
      String unresolvedUri,
      DownloadManager downloadManager,
      Map<String, String> clientEnv,
      ImmutableMap<String, Optional<Checksum>> knownFileHashes,
      KnownFileHashesMode knownFileHashesMode) {
    this.uri = uri;
    this.unresolvedUri = unresolvedUri;
    this.downloadManager = downloadManager;
    this.clientEnv = clientEnv;
    this.gson =
        new GsonBuilder()
            .setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
            .create();
    this.knownFileHashes = knownFileHashes;
    this.knownFileHashesMode = knownFileHashesMode;
  }

  @Override
  public String getUrl() {
    return unresolvedUri;
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
  private Optional<byte[]> grabFile(
      String url, ExtendedEventHandler eventHandler, boolean useChecksum)
      throws IOException, InterruptedException {
    var maybeContent = doGrabFile(url, eventHandler, useChecksum);
    if (knownFileHashesMode == KnownFileHashesMode.USE_AND_UPDATE && useChecksum) {
      eventHandler.post(RegistryFileDownloadEvent.create(url, maybeContent));
    }
    return maybeContent;
  }

  private Optional<byte[]> doGrabFile(
      String url, ExtendedEventHandler eventHandler, boolean useChecksum)
      throws IOException, InterruptedException {
    Optional<Checksum> checksum;
    if (knownFileHashesMode != KnownFileHashesMode.IGNORE && useChecksum) {
      Optional<Checksum> knownChecksum = knownFileHashes.get(url);
      if (knownChecksum == null) {
        if (knownFileHashesMode == KnownFileHashesMode.ENFORCE) {
          throw new IOException(
              String.format(
                  "Missing checksum for registry file %s. Please update the lockfile with "
                      + "`bazel mod deps --lockfile_mode=update`.",
                  url));
        }
        // This is a new file, download without providing a checksum.
        checksum = Optional.empty();
      } else if (knownChecksum.isEmpty()) {
        // The file is known to not exist, so don't attempt to download it.
        return Optional.empty();
      } else {
        // The file is known, download with a checksum to potentially obtain a repository cache hit
        // and ensure that the remote file hasn't changed.
        checksum = knownChecksum;
      }
    } else {
      checksum = Optional.empty();
    }
    if (knownFileHashesMode == KnownFileHashesMode.ENFORCE) {
      Preconditions.checkState(
          checksum.isPresent(),
          "Cannot fetch a file without a checksum in ENFORCE mode. This is a bug in Bazel, please "
              + "report at https://github.com/bazelbuild/bazel/issues/new/choose.");
    }
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.BZLMOD, () -> "download file: " + url)) {
      return Optional.of(
          downloadManager.downloadAndReadOneUrl(new URL(url), eventHandler, clientEnv, checksum));
    } catch (FileNotFoundException e) {
      return Optional.empty();
    } catch (IOException e) {
      // Include the URL in the exception message for easier debugging.
      throw new IOException(
          "Failed to fetch registry file %s: %s".formatted(url, e.getMessage()), e);
    }
  }

  @Override
  public Optional<ModuleFile> getModuleFile(ModuleKey key, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    String url =
        constructUrl(
            uri.toString(), "modules", key.getName(), key.getVersion().toString(), "MODULE.bazel");
    Optional<byte[]> maybeContent = grabFile(url, eventHandler, /* useChecksum= */ true);
    return maybeContent.map(content -> ModuleFile.create(content, url));
  }

  /** Represents fields available in {@code bazel_registry.json} for the registry. */
  private static class BazelRegistryJson {
    String[] mirrors;
    String moduleBasePath;
  }

  /** Represents the type field in {@code source.json} for each version of a module. */
  private static class SourceJson {
    String type = "archive";
  }

  /** Represents fields in {@code source.json} for each archive-type version of a module. */
  private static class ArchiveSourceJson {
    URL url;
    String integrity;
    String stripPrefix;
    Map<String, String> patches;
    int patchStrip;
    String archiveType;
  }

  /** Represents fields in {@code source.json} for each local_path-type version of a module. */
  private static class LocalPathSourceJson {
    String path;
  }

  /** Represents fields in {@code source.json} for each git_repository-type version of a module. */
  private static class GitRepoSourceJson {
    String remote;
    String commit;
    String shallowSince;
    String tag;
    boolean initSubmodules;
    boolean verbose;
    String stripPrefix;
  }

  /**
   * Grabs a JSON file from the given URL, and returns its content. Returns {@link Optional#empty}
   * if the file doesn't exist.
   */
  private Optional<String> grabJsonFile(
      String url, ExtendedEventHandler eventHandler, boolean useChecksum)
      throws IOException, InterruptedException {
    return grabFile(url, eventHandler, useChecksum).map(value -> new String(value, UTF_8));
  }

  /**
   * Grabs a JSON file from the given URL, and returns it as a parsed object with fields in {@code
   * T}. Returns {@link Optional#empty} if the file doesn't exist.
   */
  private <T> Optional<T> grabJson(
      String url, Class<T> klass, ExtendedEventHandler eventHandler, boolean useChecksum)
      throws IOException, InterruptedException {
    Optional<String> jsonString = grabJsonFile(url, eventHandler, useChecksum);
    if (jsonString.isEmpty() || jsonString.get().isBlank()) {
      return Optional.empty();
    }
    return Optional.of(parseJson(jsonString.get(), url, klass));
  }

  /** Parses the given JSON string and returns it as an object with fields in {@code T}. */
  private <T> T parseJson(String jsonString, String url, Class<T> klass) throws IOException {
    try {
      return gson.fromJson(jsonString, klass);
    } catch (JsonParseException e) {
      throw new IOException(
          String.format("Unable to parse json at url %s: %s", url, e.getMessage()), e);
    }
  }

  @Override
  public RepoSpec getRepoSpec(ModuleKey key, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    String jsonUrl =
        constructUrl(
            uri.toString(),
            "modules",
            key.getName(),
            key.getVersion().toString(),
            SOURCE_JSON_FILENAME);
    Optional<String> jsonString = grabJsonFile(jsonUrl, eventHandler, /* useChecksum= */ true);
    if (jsonString.isEmpty()) {
      throw new FileNotFoundException(
          String.format("Module %s's %s not found in registry %s", key, SOURCE_JSON_FILENAME, uri));
    }
    SourceJson sourceJson = parseJson(jsonString.get(), jsonUrl, SourceJson.class);
    switch (sourceJson.type) {
      case "archive":
        {
          ArchiveSourceJson typedSourceJson =
              parseJson(jsonString.get(), jsonUrl, ArchiveSourceJson.class);
          return createArchiveRepoSpec(typedSourceJson, getBazelRegistryJson(eventHandler), key);
        }
      case "local_path":
        {
          LocalPathSourceJson typedSourceJson =
              parseJson(jsonString.get(), jsonUrl, LocalPathSourceJson.class);
          return createLocalPathRepoSpec(typedSourceJson, getBazelRegistryJson(eventHandler), key);
        }
      case "git_repository":
        {
          GitRepoSourceJson typedSourceJson =
              parseJson(jsonString.get(), jsonUrl, GitRepoSourceJson.class);
          return createGitRepoSpec(typedSourceJson);
        }
      default:
        throw new IOException(
            String.format("Invalid source type \"%s\" for module %s", sourceJson.type, key));
    }
  }

  @SuppressWarnings("OptionalAssignedToNull")
  private Optional<BazelRegistryJson> getBazelRegistryJson(ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    if (bazelRegistryJson == null) {
      synchronized (this) {
        if (bazelRegistryJson == null) {
          var storedEventHandler = new StoredEventHandler();
          bazelRegistryJson =
              grabJson(
                  constructUrl(uri.toString(), "bazel_registry.json"),
                  BazelRegistryJson.class,
                  storedEventHandler,
                  /* useChecksum= */ true);
          bazelRegistryJsonEvents = storedEventHandler;
        }
      }
    }
    bazelRegistryJsonEvents.replayOn(eventHandler);
    return bazelRegistryJson;
  }

  private RepoSpec createLocalPathRepoSpec(
      LocalPathSourceJson sourceJson, Optional<BazelRegistryJson> bazelRegistryJson, ModuleKey key)
      throws IOException {
    String path = sourceJson.path;
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
            AttributeValues.create(ImmutableMap.of("path", PathFragment.create(path).toString())))
        .build();
  }

  private RepoSpec createArchiveRepoSpec(
      ArchiveSourceJson sourceJson, Optional<BazelRegistryJson> bazelRegistryJson, ModuleKey key)
      throws IOException {
    URL sourceUrl = sourceJson.url;
    if (sourceUrl == null) {
      throw new IOException(String.format("Missing source URL for module %s", key));
    }
    if (sourceJson.integrity == null) {
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
    if (sourceJson.patches != null) {
      for (Map.Entry<String, String> entry : sourceJson.patches.entrySet()) {
        remotePatches.put(
            constructUrl(
                unresolvedUri,
                "modules",
                key.getName(),
                key.getVersion().toString(),
                "patches",
                entry.getKey()),
            entry.getValue());
      }
    }

    return new ArchiveRepoSpecBuilder()
        .setUrls(urls.build())
        .setIntegrity(sourceJson.integrity)
        .setStripPrefix(Strings.nullToEmpty(sourceJson.stripPrefix))
        .setRemotePatches(remotePatches.buildOrThrow())
        .setRemotePatchStrip(sourceJson.patchStrip)
        .setArchiveType(sourceJson.archiveType)
        .build();
  }

  private RepoSpec createGitRepoSpec(GitRepoSourceJson sourceJson) {
    return new GitRepoSpecBuilder()
        .setRemote(sourceJson.remote)
        .setCommit(sourceJson.commit)
        .setShallowSince(sourceJson.shallowSince)
        .setTag(sourceJson.tag)
        .setInitSubmodules(sourceJson.initSubmodules)
        .setVerbose(sourceJson.verbose)
        .setStripPrefix(sourceJson.stripPrefix)
        .build();
  }

  @Override
  public Optional<ImmutableMap<Version, String>> getYankedVersions(
      String moduleName, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException {
    Optional<MetadataJson> metadataJson =
        grabJson(
            constructUrl(uri.toString(), "modules", moduleName, "metadata.json"),
            MetadataJson.class,
            eventHandler,
            // metadata.json is not immutable
            /* useChecksum= */ false);
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
