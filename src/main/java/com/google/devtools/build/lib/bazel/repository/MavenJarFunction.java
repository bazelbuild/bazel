// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
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
 * Implementation of maven_jar.
 */
public class MavenJarFunction extends HttpArchiveFunction {

  public MavenJarFunction(AtomicReference<HttpDownloader> httpDownloader) {
    super(httpDownloader);
  }

  private static final String DEFAULT_SERVER = "default";

  @Override
  public boolean isLocal(Rule rule) {
    return false;
  }

  @Override
  protected byte[] getRuleSpecificMarkerData(Rule rule, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    MavenServerValue serverValue = getServer(rule, env);
    if (env.valuesMissing()) {
      return null;
    }

    return new Fingerprint()
        .addString(serverValue.getUrl())
        .addBytes(serverValue.getSettingsFingerprint())
        .digestAndReset();
  }

  private static MavenServerValue getServer(Rule rule, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
    boolean hasRepository = mapper.isAttributeValueExplicitlySpecified("repository");
    boolean hasServer = mapper.isAttributeValueExplicitlySpecified("server");

    if (hasRepository && hasServer) {
      throw new RepositoryFunctionException(new EvalException(
          rule.getLocation(), rule + " specifies both "
          + "'repository' and 'server', which are mutually exclusive options"),
          Transience.PERSISTENT);
    }

    try {
      if (hasRepository) {
        return MavenServerValue.createFromUrl(mapper.get("repository", Type.STRING));
      } else {
        String serverName = DEFAULT_SERVER;
        if (hasServer) {
          serverName = mapper.get("server", Type.STRING);
        }
        return (MavenServerValue) env.getValue(MavenServerValue.key(serverName));
      }
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }

  }

  @Override
  public SkyValue fetch(
      Rule rule, Path outputDirectory, BlazeDirectories directories, Environment env)
          throws RepositoryFunctionException, InterruptedException {
    MavenServerValue serverValue = getServer(rule, env);
    if (env.valuesMissing()) {
      return null;
    }
    MavenDownloader downloader;
    try {
      downloader = createMavenDownloader(directories, rule, serverValue);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }
    return createOutputTree(downloader);
  }

  private MavenDownloader createMavenDownloader(
      BlazeDirectories directories, Rule rule, MavenServerValue serverValue)
      throws IOException, EvalException {
    String name = rule.getName();
    Path outputDirectory = getExternalRepositoryDirectory(directories).getRelative(name);
    return new MavenDownloader(
        name, WorkspaceAttributeMapper.of(rule), outputDirectory, serverValue);
  }

  private SkyValue createOutputTree(MavenDownloader downloader)
      throws RepositoryFunctionException, InterruptedException {
    Path outputDirectory = downloader.getOutputDirectory();
    createDirectory(outputDirectory);
    Path repositoryJar;

    try {
      repositoryJar = downloader.download();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    // Add a WORKSPACE file & BUILD file to the Maven jar.
    Path result = DecompressorValue.decompress(DecompressorDescriptor.builder()
        .setDecompressor(JarDecompressor.INSTANCE)
        .setTargetKind(MavenJarRule.NAME)
        .setTargetName(downloader.getName())
        .setArchivePath(repositoryJar)
        .setRepositoryPath(outputDirectory).build());
    return RepositoryDirectoryValue.create(result);
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
    private final Server server;

    public MavenDownloader(
        String name, WorkspaceAttributeMapper mapper, Path outputDirectory,
        MavenServerValue serverValue)
        throws IOException, EvalException {
      this.name = name;
      this.outputDirectory = outputDirectory;

      this.artifact = mapper.get("artifact", Type.STRING);
      this.sha1 = mapper.isAttributeValueExplicitlySpecified("sha1")
          ? mapper.get("sha1", Type.STRING) : null;
      if (sha1 != null && !sha1.matches("\\p{XDigit}{40}")) {
        throw new IOException("Invalid SHA-1 for maven_jar " + name + ": '" + sha1 + "'");
      }
      this.url = serverValue.getUrl();
      this.server = serverValue.getServer();
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


      RemoteRepository repository = new RemoteRepository.Builder(
          name, MavenServerValue.DEFAULT_ID, url)
          .setAuthentication(new MavenAuthentication(server))
          .build();
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
        RepositoryCache.assertFileChecksum(sha1, downloadPath, KeyType.SHA1);
      }
      return downloadPath;
    }
  }

  private static class MavenAuthentication implements Authentication {

    private final Map<String, String> authenticationInfo;

    private MavenAuthentication(Server server) {
      ImmutableMap.Builder builder = ImmutableMap.<String, String>builder();
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
