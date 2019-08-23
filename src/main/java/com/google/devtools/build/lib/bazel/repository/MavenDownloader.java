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
import com.google.devtools.build.lib.events.Location;
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
      Location location,
      WorkspaceAttributeMapper mapper,
      Path outputDirectory,
      MavenServerValue serverValue,
      ExtendedEventHandler eventHandler)
      throws IOException, EvalException, InterruptedException {

    String url = serverValue.getUrl();
    Server server = serverValue.getServer();

    // Initialize maven artifacts
    String artifactCoords = mapper.get("artifact", Type.STRING);

    KeyType keyType = KeyType.SHA256;
    String checksum = retrieveChecksum(name, "sha256", keyType, mapper);

    KeyType srcJarKeyType = KeyType.SHA256;
    String srcJarChecksum = retrieveChecksum(name, "sha256_src", srcJarKeyType, mapper);

    if (checksum == null) {
      keyType = KeyType.SHA1;
      checksum = retrieveChecksum(name, "sha1", keyType, mapper);
    }

    if (srcJarChecksum == null) {
      srcJarKeyType = KeyType.SHA1;
      srcJarChecksum = retrieveChecksum(name, "sha1_src", keyType, mapper);
    }

    Artifact artifact;
    try {
      artifact = new DefaultArtifact(artifactCoords);
    } catch (IllegalArgumentException e) {
      throw new IOException(e.getMessage());
    }

    Artifact artifactWithSrcs = srcjarCoords(artifact);

    boolean isCaching = repositoryCache.isEnabled() && keyType.isValid(checksum);

    if (isCaching) {
      Path downloadPath = getDownloadDestination(outputDirectory, artifact);
      try {
        Path cachedDestination = repositoryCache.get(checksum, downloadPath, keyType);
        if (cachedDestination != null) {
          Path cachedDestinationSrc = null;
          if (srcJarChecksum != null) {
            Path downloadPathSrc = getDownloadDestination(outputDirectory, artifactWithSrcs);
            cachedDestinationSrc =
                repositoryCache.get(srcJarChecksum, downloadPathSrc, srcJarKeyType);
          }
          return new JarPaths(cachedDestination, Optional.fromNullable(cachedDestinationSrc));
        }
      } catch (IOException e) {
        eventHandler.handle(
            Event.debug("RepositoryCache entry " + checksum + " is invalid, replacing it..."));
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
    if (!Strings.isNullOrEmpty(checksum)) {
      RepositoryCache.assertFileChecksum(checksum, jarDownload, keyType);
    }

    Path srcjarDownload = null;
    if (artifactWithSrcs.getFile() != null) {
      srcjarDownload = outputDirectory.getRelative(artifactWithSrcs.getFile().getAbsolutePath());
      if (!Strings.isNullOrEmpty(srcJarChecksum)) {
        RepositoryCache.assertFileChecksum(srcJarChecksum, srcjarDownload, srcJarKeyType);
      }
    }

    if (isCaching) {
      repositoryCache.put(checksum, jarDownload, keyType);
      if (srcjarDownload != null && !Strings.isNullOrEmpty(srcJarChecksum)) {
        repositoryCache.put(srcJarChecksum, srcjarDownload, srcJarKeyType);
      }
    }

    // The detected keytype is not SHA-256. This is a security risk because missing checksums mean
    // there's no integrity checking and SHA-1 is cryptographically insecure. Let's be helpful and
    // print out the computed SHA-256 from the downloaded jar(s).
    if (keyType != KeyType.SHA256) {
      String commonMessage =
          String.format(
              "maven_jar rule @%s//jar: Not using a checksum to verify the integrity of "
                  + "the artifact or the usage of SHA-1 is not secure (see https://shattered.io) and "
                  + "can result in an non-reproducible build.",
              name);

      String warningMessage =
          String.format(
              commonMessage + " Please specify the SHA-256 checksum with: sha256 = \"%s\",",
              RepositoryCache.getChecksum(KeyType.SHA256, jarDownload));

      eventHandler.handle(Event.warn(location, warningMessage));

      if (srcjarDownload != null && srcJarKeyType != KeyType.SHA256) {
        warningMessage =
            String.format(
                commonMessage + " Please specify the SHA-256 checksum with: sha256_src = \"%s\",",
                RepositoryCache.getChecksum(KeyType.SHA256, srcjarDownload));

        eventHandler.handle(Event.warn(location, warningMessage));
      }
    }

    return new JarPaths(jarDownload, Optional.fromNullable(srcjarDownload));
  }

  @Nullable
  private String retrieveChecksum(
      String name, String attribute, KeyType keyType, WorkspaceAttributeMapper mapper)
      throws EvalException, IOException {
    String checksum =
        mapper.isAttributeValueExplicitlySpecified(attribute)
            ? mapper.get(attribute, Type.STRING)
            : null;
    if (checksum != null && !keyType.isValid(checksum)) {
      throw new IOException(
          "Invalid " + keyType.toString()+ " for maven_jar " + name + ": '" + checksum + "'");
    }
    return checksum;
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
