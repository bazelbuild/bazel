// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenServerRule;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.ExternalPackage;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.apache.maven.settings.Server;
import org.apache.maven.settings.Settings;
import org.apache.maven.settings.building.DefaultSettingsBuilder;
import org.apache.maven.settings.building.DefaultSettingsBuilderFactory;
import org.apache.maven.settings.building.DefaultSettingsBuildingRequest;
import org.apache.maven.settings.building.SettingsBuildingException;
import org.apache.maven.settings.building.SettingsBuildingResult;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.annotation.Nullable;

/**
 * Implementation of maven_repository.
 */
public class MavenServerFunction extends RepositoryFunction {
  public static final SkyFunctionName NAME = SkyFunctionName.create("MAVEN_SERVER_FUNCTION");

  public MavenServerFunction(BlazeDirectories directories) {
    setDirectories(directories);
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws RepositoryFunctionException {
    String repository = skyKey.argument().toString();
    ExternalPackage externalPackage = RepositoryFunction.getExternalPackage(env);
    Rule repositoryRule = externalPackage.getRule(repository);

    boolean foundRepoRule = repositoryRule != null
        && repositoryRule.getRuleClass().equals(MavenServerRule.NAME);
    if (!foundRepoRule) {
      if (repository.equals(MavenServerValue.DEFAULT_ID)) {
        // The default repository is being used and the WORKSPACE is not overriding the default.
        return new MavenServerValue();
      }
      throw new RepositoryFunctionException(
          new IOException("Could not find maven repository " + repository), Transience.TRANSIENT);
    }

    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(repositoryRule);
    String serverName = repositoryRule.getName();
    String url = mapper.get("url", Type.STRING);
    if (!mapper.has("settings_file", Type.STRING)
        || mapper.get("settings_file", Type.STRING).isEmpty()) {
      return new MavenServerValue(serverName, url, new Server());
    }
    PathFragment settingsFilePath = new PathFragment(mapper.get("settings_file", Type.STRING));
    RootedPath settingsPath = RootedPath.toRootedPath(
        getWorkspace().getRelative(settingsFilePath), PathFragment.EMPTY_FRAGMENT);
    FileValue settingsFile = (FileValue) env.getValue(FileValue.key(settingsPath));
    if (settingsFile == null) {
      return null;
    }

    if (!settingsFile.exists()) {
      throw new RepositoryFunctionException(
          new IOException("Could not find settings file " + settingsPath), Transience.TRANSIENT);
    }

    DefaultSettingsBuildingRequest request = new DefaultSettingsBuildingRequest();
    request.setUserSettingsFile(new File(settingsFile.realRootedPath().asPath().toString()));
    DefaultSettingsBuilder builder = (new DefaultSettingsBuilderFactory()).newInstance();
    SettingsBuildingResult result;
    try {
      result = builder.build(request);
    } catch (SettingsBuildingException e) {
      throw new RepositoryFunctionException(
          new IOException("Error parsing settings file " + settingsFile + ": " + e.getMessage()),
          Transience.TRANSIENT);
    }
    if (!result.getProblems().isEmpty()) {
      throw new RepositoryFunctionException(
          new IOException("Errors interpreting settings file: "
              + Arrays.toString(result.getProblems().toArray())), Transience.PERSISTENT);
    }
    Settings settings = result.getEffectiveSettings();
    Server server = settings.getServer(mapper.getName());
    server = server == null ? new Server() : server;
    return new MavenServerValue(serverName, url, server);
  }

  @Override
  public SkyFunctionName getSkyFunctionName() {
    return NAME;
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return MavenServerRule.class;
  }
}
