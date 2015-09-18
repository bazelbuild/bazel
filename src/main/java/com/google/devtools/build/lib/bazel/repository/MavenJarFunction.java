// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.eclipse.aether.RepositorySystem;
import org.eclipse.aether.RepositorySystemSession;
import org.eclipse.aether.artifact.Artifact;
import org.eclipse.aether.artifact.DefaultArtifact;
import org.eclipse.aether.repository.RemoteRepository;
import org.eclipse.aether.resolution.ArtifactRequest;
import org.eclipse.aether.resolution.ArtifactResolutionException;
import org.eclipse.aether.resolution.ArtifactResult;

import java.io.IOException;

import javax.annotation.Nullable;

/**
 * Implementation of maven_jar.
 */
public class MavenJarFunction extends HttpArchiveFunction {

  private static final String DEFAULT_SERVER = "default";

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws RepositoryFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(repositoryName, MavenJarRule.NAME, env);
    if (rule == null) {
      return null;
    }

    String url;
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    boolean hasRepository = mapper.has("repository", Type.STRING)
        && !mapper.get("repository", Type.STRING).isEmpty();
    boolean hasServer = mapper.has("server", Type.STRING)
        && !mapper.get("server", Type.STRING).isEmpty();
    if (hasRepository && hasServer) {
      throw new RepositoryFunctionException(new EvalException(
          Location.fromFile(getWorkspace().getRelative("WORKSPACE")), rule + " specifies both "
              + "'repository' and 'server', which are mutually exclusive options"),
          Transience.PERSISTENT);
    } else if (hasRepository) {
      url = mapper.get("repository", Type.STRING);
    } else {
      String serverName = DEFAULT_SERVER;
      if (mapper.has("server", Type.STRING) && !mapper.get("server", Type.STRING).isEmpty()) {
        serverName = mapper.get("server", Type.STRING);
      }

      MavenServerValue mavenServerValue = (MavenServerValue) env.getValue(
          MavenServerValue.key(serverName));
      if (mavenServerValue == null) {
        return null;
      }
      url = mavenServerValue.getUrl();
    }

    MavenDownloader downloader = createMavenDownloader(mapper, url);
    return createOutputTree(downloader, env);
  }

  @VisibleForTesting
  MavenDownloader createMavenDownloader(AttributeMap mapper, String url) {
    String name = mapper.getName();
    Path outputDirectory = getExternalRepositoryDirectory().getRelative(name);
    return new MavenDownloader(name, mapper, outputDirectory, url);
  }

  SkyValue createOutputTree(MavenDownloader downloader, Environment env)
      throws RepositoryFunctionException {
    FileValue outputDirectoryValue = createDirectory(downloader.getOutputDirectory(), env);
    if (outputDirectoryValue == null) {
      return null;
    }

    Path repositoryJar;
    try {
      repositoryJar = downloader.download();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    // Add a WORKSPACE file & BUILD file to the Maven jar.
    DecompressorValue value;
    try {
      value = (DecompressorValue) env.getValueOrThrow(DecompressorValue.jarKey(
          MavenJarRule.NAME, downloader.getName(), repositoryJar,
          outputDirectoryValue.realRootedPath().asPath()), IOException.class);
      if (value == null) {
        return null;
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    FileValue repositoryFileValue = getRepositoryDirectory(value.getDirectory(), env);
    if (repositoryFileValue == null) {
      return null;
    }
    return RepositoryValue.create(repositoryFileValue);
  }

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.create(Ascii.toUpperCase(MavenJarRule.NAME));
  }

  /**
   * @see RepositoryFunction#getRule(RepositoryName, String, Environment)
   */
  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return MavenJarRule.class;
  }

  /**
   * This downloader creates a connection to one or more Maven repositories and downloads a jar.
   */
  static class MavenDownloader {
    private final String name;
    private final String artifact;
    private final Path outputDirectory;
    @Nullable
    private final String sha1;
    private final String url;

    public MavenDownloader(
        String name, AttributeMap mapper, Path outputDirectory, String url) {
      this.name = name;
      this.outputDirectory = outputDirectory;

      if (!mapper.get("artifact", Type.STRING).isEmpty()) {
        this.artifact = mapper.get("artifact", Type.STRING);
      } else {
        this.artifact = mapper.get("group_id", Type.STRING) + ":"
            + mapper.get("artifact_id", Type.STRING) + ":"
            + mapper.get("version", Type.STRING);
      }
      this.sha1 = (mapper.has("sha1", Type.STRING)) ? mapper.get("sha1", Type.STRING) : null;
      this.url = url;
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
    public Path download() throws IOException {
      MavenConnector connector = new MavenConnector(outputDirectory.getPathString());
      RepositorySystem system = connector.newRepositorySystem();
      RepositorySystemSession session = connector.newRepositorySystemSession(system);

      RemoteRepository repository = new RemoteRepository.Builder(name, "default", url).build();
      ArtifactRequest artifactRequest = new ArtifactRequest();
      Artifact artifact;
      try {
        artifact = new DefaultArtifact(this.artifact);
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
        Hasher hasher = Hashing.sha1().newHasher();
        String downloadSha1 = HttpDownloader.getHash(hasher, downloadPath);
        if (!sha1.equals(downloadSha1)) {
          throw new IOException("Downloaded file at " + downloadPath + " has SHA-1 of "
              + downloadSha1 + ", does not match expected SHA-1 (" + sha1 + ")");
        }
      }
      return downloadPath;
    }
  }

}
