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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.repository.DecompressorFactory.DecompressorException;
import com.google.devtools.build.lib.bazel.repository.DecompressorFactory.JarDecompressor;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.apache.maven.repository.internal.MavenRepositorySystemUtils;
import org.eclipse.aether.AbstractRepositoryListener;
import org.eclipse.aether.DefaultRepositorySystemSession;
import org.eclipse.aether.RepositorySystem;
import org.eclipse.aether.RepositorySystemSession;
import org.eclipse.aether.artifact.Artifact;
import org.eclipse.aether.artifact.DefaultArtifact;
import org.eclipse.aether.connector.basic.BasicRepositoryConnectorFactory;
import org.eclipse.aether.impl.DefaultServiceLocator;
import org.eclipse.aether.repository.LocalRepository;
import org.eclipse.aether.repository.RemoteRepository;
import org.eclipse.aether.resolution.ArtifactRequest;
import org.eclipse.aether.resolution.ArtifactResolutionException;
import org.eclipse.aether.resolution.ArtifactResult;
import org.eclipse.aether.spi.connector.RepositoryConnectorFactory;
import org.eclipse.aether.spi.connector.transport.TransporterFactory;
import org.eclipse.aether.transfer.AbstractTransferListener;
import org.eclipse.aether.transport.file.FileTransporterFactory;
import org.eclipse.aether.transport.http.HttpTransporterFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Implementation of maven_jar.
 */
public class MavenJarFunction extends HttpArchiveFunction {

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws RepositoryFunctionException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();
    Rule rule = RepositoryFunction.getRule(repositoryName, MavenJarRule.NAME, env);
    if (rule == null) {
      return null;
    }
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    MavenDownloader downloader = createMavenDownloader(mapper);
    return createOutputTree(downloader, env);
  }

  @VisibleForTesting
  MavenDownloader createMavenDownloader(AttributeMap mapper) {
    String name = mapper.getName();
    Path outputDirectory = getExternalRepositoryDirectory().getRelative(name);
    MavenDownloader downloader = new MavenDownloader(name, mapper.get("group_id", Type.STRING),
        mapper.get("artifact_id", Type.STRING), mapper.get("version", Type.STRING),
        outputDirectory);

    List<String> repositories = mapper.get("repositories", Type.STRING_LIST);
    if (repositories != null && !repositories.isEmpty()) {
      downloader.setRepositories(repositories);
    }

    return downloader;
  }

  @VisibleForTesting
  SkyValue createOutputTree(MavenDownloader downloader, Environment env)
      throws RepositoryFunctionException {

    FileValue outputDirectoryValue = createDirectory(downloader.getOutputDirectory(), env);
    if (outputDirectoryValue == null) {
      return null;
    }

    Path repositoryJar = null;
    try {
      repositoryJar = downloader.download();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    // Add a WORKSPACE file & BUILD file to the Maven jar.
    JarDecompressor decompressor = new JarDecompressor(
        MavenJarRule.NAME, downloader.getName(), repositoryJar,
        outputDirectoryValue.realRootedPath().asPath());
    Path repositoryDirectory = null;
    try {
      repositoryDirectory = decompressor.decompress();
    } catch (DecompressorException e) {
      throw new RepositoryFunctionException(new IOException(e.getMessage()), Transience.TRANSIENT);
    }
    FileValue repositoryFileValue = getRepositoryDirectory(repositoryDirectory, env);
    if (repositoryFileValue == null) {
      return null;
    }
    return new RepositoryValue(repositoryDirectory, repositoryFileValue);
  }

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return SkyFunctionName.computed(Ascii.toUpperCase(MavenJarRule.NAME));
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return MavenJarRule.class;
  }

  static class MavenDownloader {
    private static final String MAVEN_CENTRAL_URL = "http://central.maven.org/maven2/";

    private final String name;
    private final String groupId;
    private final String artifactId;
    private final String version;
    private final Path outputDirectory;
    private List<RemoteRepository> repositories;

    MavenDownloader(String name, String groupId, String artifactId, String version,
        Path outputDirectory) {
      this.name = name;
      this.groupId = groupId;
      this.artifactId = artifactId;
      this.version = version;
      this.outputDirectory = outputDirectory;
      repositories = Arrays.asList(
          new RemoteRepository.Builder("central", "default", MAVEN_CENTRAL_URL).build());
      this.repositories = ImmutableList.copyOf(repositories);
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
     * Customizes the set of Maven repositories to check.  Takes a list of repository URLs.
     */
    public void setRepositories(List<String> repositories) {
      this.repositories = Lists.newArrayList();
      for (String repositoryUrl : repositories) {
        this.repositories.add(new RemoteRepository.Builder(
            "user-defined repository " + repositories.size(), "default", repositoryUrl).build());
      }
    }

    /**
     * Download the Maven artifact to the output directory. Returns the path to the jar.
     */
    public Path download() throws IOException {
      RepositorySystem system = newRepositorySystem();
      RepositorySystemSession session = newRepositorySystemSession(system);

      ArtifactRequest artifactRequest = new ArtifactRequest();
      Artifact artifact = new DefaultArtifact(groupId + ":" + artifactId + ":" + version);
      artifactRequest.setArtifact(artifact);
      artifactRequest.setRepositories(repositories);

      try {
        ArtifactResult artifactResult = system.resolveArtifact(session, artifactRequest);
        artifact = artifactResult.getArtifact();
      } catch (ArtifactResolutionException e) {
        throw new IOException("Failed to fetch Maven dependency: " + e.getMessage());
      }
      return outputDirectory.getRelative(artifact.getFile().getAbsolutePath());
    }

    private RepositorySystemSession newRepositorySystemSession(RepositorySystem system) {
      DefaultRepositorySystemSession session = MavenRepositorySystemUtils.newSession();
      LocalRepository localRepo = new LocalRepository(outputDirectory.getPathString());
      session.setLocalRepositoryManager(system.newLocalRepositoryManager(session, localRepo));
      session.setTransferListener(new AbstractTransferListener() {});
      session.setRepositoryListener(new AbstractRepositoryListener() {});
      return session;
    }

    private RepositorySystem newRepositorySystem() {
      DefaultServiceLocator locator = MavenRepositorySystemUtils.newServiceLocator();
      locator.addService(RepositoryConnectorFactory.class, BasicRepositoryConnectorFactory.class);
      locator.addService(TransporterFactory.class, FileTransporterFactory.class);
      locator.addService(TransporterFactory.class, HttpTransporterFactory.class);
      return locator.getService(RepositorySystem.class);
    }
  }

}
