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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.util.Map;

/**
 * Implementation of maven_jar.
 */
public class MavenJarFunction extends HttpArchiveFunction {

  public MavenJarFunction(MavenDownloader mavenDownloader) {
    super(mavenDownloader);
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
  public RepositoryDirectoryValue.Builder fetch(Rule rule, Path outputDirectory,
      BlazeDirectories directories, Environment env, Map<String, String> markerData)
      throws RepositoryFunctionException, InterruptedException {
    MavenServerValue serverValue = getServer(rule, env);
    if (env.valuesMissing()) {
      return null;
    }
    
    Path outputDir = getExternalRepositoryDirectory(directories).getRelative(rule.getName());
    return createOutputTree(rule, outputDir, serverValue);
  }

  private RepositoryDirectoryValue.Builder createOutputTree(Rule rule, Path outputDirectory,
      MavenServerValue serverValue) throws RepositoryFunctionException, InterruptedException {
    Preconditions.checkState(downloader instanceof MavenDownloader);
    MavenDownloader mavenDownloader = (MavenDownloader) downloader;

    createDirectory(outputDirectory);
    String name = rule.getName();
    Path repositoryJar;
    try {
      repositoryJar = mavenDownloader.download(
          name, WorkspaceAttributeMapper.of(rule), outputDirectory, serverValue);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    // Add a WORKSPACE file & BUILD file to the Maven jar.
    Path result = DecompressorValue.decompress(DecompressorDescriptor.builder()
        .setDecompressor(JarDecompressor.INSTANCE)
        .setTargetKind(MavenJarRule.NAME)
        .setTargetName(name)
        .setArchivePath(repositoryJar)
        .setRepositoryPath(outputDirectory).build());
    return RepositoryDirectoryValue.builder().setPath(result);
  }

  /**
   * @see RepositoryFunction#getRule(RepositoryName, String, Environment)
   */
  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return MavenJarRule.class;
  }

}
