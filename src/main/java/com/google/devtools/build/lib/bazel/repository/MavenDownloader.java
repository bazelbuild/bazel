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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Map;
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

  @Nullable
  private String name;
  @Nullable
  private Path outputDirectory;

  public MavenDownloader(RepositoryCache repositoryCache) {
    super(repositoryCache);
  }

  /**
   * Returns the name for this artifact-fetching rule.
   */
  public String getName() {
    return name;
  }

  /**
   * Returns the directory that this artifact will be downloaded to.
   */
  public Path getOutputDirectory() {
    return outputDirectory;
  }

  /**
   * Download the Maven artifact to the output directory. Returns the path to the jar.
   */
  public Path download(String name, WorkspaceAttributeMapper mapper, Path outputDirectory,
      MavenServerValue serverValue) throws IOException, EvalException {
    this.name = name;
    this.outputDirectory = outputDirectory;
    String artifactId = mapper.get("artifact", Type.STRING);
    String sha1 = mapper.isAttributeValueExplicitlySpecified("sha1")
        ? mapper.get("sha1", Type.STRING) : null;
        if (sha1 != null && !KeyType.SHA1.isValid(sha1)) {
          throw new IOException("Invalid SHA-1 for maven_jar " + name + ": '" + sha1 + "'");
        }
    String url = serverValue.getUrl();
    Server server = serverValue.getServer();

    MavenConnector connector = new MavenConnector(outputDirectory.getPathString());
    RepositorySystem system = connector.newRepositorySystem();
    RepositorySystemSession session = connector.newRepositorySystemSession(system);

    RemoteRepository repository = new RemoteRepository.Builder(
        name, MavenServerValue.DEFAULT_ID, url)
        .setAuthentication(new MavenAuthentication(server))
        .build();
    ArtifactRequest artifactRequest = new ArtifactRequest();
    Artifact artifact;
    try {
      artifact = new DefaultArtifact(artifactId);
    } catch (IllegalArgumentException e) {
      throw new IOException(e.getMessage());
    }
    artifactRequest.setArtifact(artifact);
    artifactRequest.setRepositories(ImmutableList.of(repository));

    try {
      ArtifactResult artifactResult = system.resolveArtifact(session, artifactRequest);
      artifact = artifactResult.getArtifact();
    } catch (ArtifactResolutionException e) {
      throw new IOException("Failed to fetch Maven dependency: " + e.getMessage());
    }

    Path downloadPath = outputDirectory.getRelative(artifact.getFile().getAbsolutePath());
    // Verify checksum.
    if (!Strings.isNullOrEmpty(sha1)) {
      RepositoryCache.assertFileChecksum(sha1, downloadPath, KeyType.SHA1);
    }
    return downloadPath;
  }

  private static class MavenAuthentication implements Authentication {

    private final Map<String, String> authenticationInfo;

    private MavenAuthentication(Server server) {
      Builder<String, String> builder = ImmutableMap.<String, String>builder();
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
