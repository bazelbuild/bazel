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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Map;
import java.util.StringJoiner;
import javax.annotation.Nullable;
import org.apache.maven.settings.Server;
import org.eclipse.aether.RepositorySystem;
import org.eclipse.aether.RepositorySystemSession;
import org.eclipse.aether.artifact.Artifact;
import org.eclipse.aether.artifact.DefaultArtifact;
import org.eclipse.aether.repository.Authentication;
import org.eclipse.aether.repository.AuthenticationContext;
import org.eclipse.aether.repository.AuthenticationDigest;
import org.eclipse.aether.repository.RemoteRepository;
import org.eclipse.aether.resolution.ArtifactRequest;
import org.eclipse.aether.resolution.ArtifactResolutionException;
import org.eclipse.aether.resolution.ArtifactResult;

/**
 * Downloader for JAR files from Maven repositories.
 * TODO(jingwen): standardize interface between this and HttpDownloader
 */
public class MavenDownloader extends HttpDownloader {

  public MavenDownloader(RepositoryCache repositoryCache) {
    super(repositoryCache);
  }

  /**
   * Download the Maven artifact to the output directory. Returns the path to the jar (and the
   * srcjar if available).
   */
  public JarPaths download(
      String name,
      WorkspaceAttributeMapper mapper,
      Path outputDirectory,
      MavenServerValue serverValue,
      ExtendedEventHandler eventHandler)
      throws IOException, EvalException {

    String url = serverValue.getUrl();
    Server server = serverValue.getServer();

    // Initialize maven artifacts
    String artifactCoords = mapper.get("artifact", Type.STRING);
    String sha1 = retrieveSha1(name, "sha1", mapper);
    String sha1Src = retrieveSha1(name, "sha1_src", mapper);

    Artifact artifact;
    try {
      artifact = new DefaultArtifact(artifactCoords);
    } catch (IllegalArgumentException e) {
      throw new IOException(e.getMessage());
    }

    Artifact artifactWithSrcs = srcjarCoords(artifact);

    String artifactCacheKey = getCacheKey(artifact, sha1);
    boolean isCaching =
        repositoryCache.isEnabled() && KeyType.ID_PREFIXED_SHA1.isValid(artifactCacheKey);

    String srcArtifactCacheKey = null;
    if (isCaching) {
      Path downloadPath = getDownloadDestination(outputDirectory, artifact);
      try {
        Path cachedDestination =
            repositoryCache.get(artifactCacheKey, downloadPath, KeyType.ID_PREFIXED_SHA1);
        if (cachedDestination != null) {
          Path cachedDestinationSrc = null;
          if (sha1Src != null) {
            srcArtifactCacheKey = getCacheKey(artifactWithSrcs, sha1Src);
            Path downloadPathSrc = getDownloadDestination(outputDirectory, artifactWithSrcs);
            cachedDestinationSrc =
                repositoryCache.get(srcArtifactCacheKey, downloadPathSrc, KeyType.ID_PREFIXED_SHA1);
          }
          return new JarPaths(cachedDestination, Optional.fromNullable(cachedDestinationSrc));
        }
      } catch (IOException e) {
        eventHandler.handle(
            Event.debug("RepositoryCache entry " + sha1 + " is invalid, replacing it..."));
        // Ignore error trying to get. We'll just download again.
      }
    }

    // Setup env for fetching jars
    MavenConnector connector = new MavenConnector(outputDirectory.getPathString());
    RepositorySystem system = connector.newRepositorySystem();
    RepositorySystemSession session = connector.newRepositorySystemSession(system);
    RemoteRepository repository =
        new RemoteRepository.Builder(name, MavenServerValue.DEFAULT_ID, url)
            .setAuthentication(new MavenAuthentication(server))
            .build();

    // Try fetching jar.
    final Path jarDownload;
    try {
      artifact = downloadArtifact(artifact, repository, session, system);
    } catch (ArtifactResolutionException e) {
      throw new IOException("Failed to fetch Maven dependency: " + e.getMessage());
    }

    // Try also fetching srcjar.
    try {
      artifactWithSrcs = downloadArtifact(artifactWithSrcs, repository, session, system);
    } catch (ArtifactResolutionException e) {
      // Intentionally ignored - missing srcjar is not an error.
    }

    jarDownload = outputDirectory.getRelative(artifact.getFile().getAbsolutePath());
    // Verify checksum.
    if (!Strings.isNullOrEmpty(sha1)) {
      RepositoryCache.assertFileChecksum(sha1, jarDownload, KeyType.SHA1);
    }

    Path srcjarDownload = null;
    if (artifactWithSrcs.getFile() != null) {
      srcjarDownload = outputDirectory.getRelative(artifactWithSrcs.getFile().getAbsolutePath());
      if (!Strings.isNullOrEmpty(sha1Src)) {
        RepositoryCache.assertFileChecksum(sha1Src, srcjarDownload, KeyType.SHA1);
      }
    }

    if (isCaching) {
      repositoryCache.put(artifactCacheKey, jarDownload, KeyType.ID_PREFIXED_SHA1);
      if (srcjarDownload != null && !Strings.isNullOrEmpty(srcArtifactCacheKey)) {
        repositoryCache.put(srcArtifactCacheKey, srcjarDownload, KeyType.ID_PREFIXED_SHA1);
      }
    }

    return new JarPaths(jarDownload, Optional.fromNullable(srcjarDownload));
  }

  private String getCacheKey(Artifact artifact, @Nullable String sha1) {
    if (sha1 == null) {
      return null;
    }
    StringJoiner j = new StringJoiner("-")
        .add(artifact.getGroupId())
        .add(artifact.getArtifactId());
    String classifier = artifact.getClassifier();
    if (!Strings.isNullOrEmpty(classifier)) {
      j.add(classifier);
    }
    j.add(artifact.getVersion());
    j.add(sha1);
    return j.toString();
  }

  private String retrieveSha1(String name, String attribute, WorkspaceAttributeMapper mapper)
      throws EvalException, IOException {
    String sha1 =
        mapper.isAttributeValueExplicitlySpecified(attribute)
            ? mapper.get(attribute, Type.STRING)
            : null;
    if (sha1 != null && !KeyType.SHA1.isValid(sha1)) {
      throw new IOException("Invalid SHA-1 for maven_jar " + name + ": '" + sha1 + "'");
    }
    return sha1;
  }

  private Path getDownloadDestination(Path outputDirectory, Artifact artifact) {
    String groupIdPath = artifact.getGroupId().replace('.', '/');
    String artifactId = artifact.getArtifactId();
    String classifier = artifact.getClassifier();
    String version = artifact.getVersion();
    String filename = artifactId + '-' + version;

    if (classifier.equals("sources")) {
      filename += "-sources";
    }
    filename += '.' + artifact.getExtension();

    StringJoiner joiner = new StringJoiner("/");
    joiner.add(groupIdPath).add(artifactId).add(version).add(filename);

    return outputDirectory.getRelative(joiner.toString());
  }

  private Artifact srcjarCoords(Artifact jar) {
    return new DefaultArtifact(
        jar.getGroupId(), jar.getArtifactId(), "sources", jar.getExtension(), jar.getVersion());
  }

  /*
   * Set up request for and resolve (retrieve to local repo) artifact
   */
  private Artifact downloadArtifact(
      Artifact artifact,
      RemoteRepository repository,
      RepositorySystemSession session,
      RepositorySystem system)
      throws ArtifactResolutionException {
    ArtifactRequest artifactRequest = new ArtifactRequest();
    artifactRequest.setArtifact(artifact);
    artifactRequest.setRepositories(ImmutableList.of(repository));
    ArtifactResult artifactResult = system.resolveArtifact(session, artifactRequest);
    return artifactResult.getArtifact();
  }

  /*
   * Class for packaging srcjar and jar paths together when srcjar is available.
   */
  static class JarPaths {
    final Path jar;
    @Nullable final Optional<Path> srcjar;

    private JarPaths(Path jar, Optional<Path> srcjar) {
      this.jar = jar;
      this.srcjar = srcjar;
    }
  }

  private static class MavenAuthentication implements Authentication {

    private final Map<String, String> authenticationInfo;

    private MavenAuthentication(Server server) {
      ImmutableMap.Builder<String, String> builder = ImmutableMap.<String, String>builder();
      // From https://maven.apache.org/settings.html: "If you use a private key to login to the
      // server, make sure you omit the <password> element. Otherwise, the key will be ignored."
      if (server.getPassword() != null) {
        builder.put(AuthenticationContext.USERNAME, server.getUsername());
        builder.put(AuthenticationContext.PASSWORD, server.getPassword());
      } else if (server.getPrivateKey() != null) {
        // getPrivateKey sounds like it returns the key, but it actually returns a path to it.
        builder.put(AuthenticationContext.PRIVATE_KEY_PATH, server.getPrivateKey());
        builder.put(AuthenticationContext.PRIVATE_KEY_PASSPHRASE, server.getPassphrase());
      }
      authenticationInfo = builder.build();
    }

    @Override
    public void fill(
        AuthenticationContext authenticationContext, String s, Map<String, String> map) {
      for (Map.Entry<String, String> entry : authenticationInfo.entrySet()) {
        authenticationContext.put(entry.getKey(), entry.getValue());
      }
    }

    @Override
    public void digest(AuthenticationDigest authenticationDigest) {
      // No-op.
    }
  }
}
