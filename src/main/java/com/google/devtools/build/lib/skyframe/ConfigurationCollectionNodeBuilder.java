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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.view.config.BuildConfigurationKey;
import com.google.devtools.build.lib.view.config.ConfigurationFactory;
import com.google.devtools.build.lib.view.config.InvalidConfigurationException;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

/**
 * A builder for {@link ConfigurationCollectionNode}s.
 */
public class ConfigurationCollectionNodeBuilder implements NodeBuilder {

  private final Supplier<ConfigurationFactory> configurationFactory;
  private final Supplier<BuildConfigurationKey> configurationKey;
  private final Reporter reporter;

  public ConfigurationCollectionNodeBuilder(
      Supplier<ConfigurationFactory> configurationFactory,
      Supplier<BuildConfigurationKey> key,
      Reporter reporter) {
    this.configurationFactory = configurationFactory;
    this.configurationKey = key;
    this.reporter = reporter;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws InterruptedException,
      ConfigurationCollectionNodeBuilderException {
    BuildConfigurationCollection collection;
    try {
      // We are not using these values, because we have copies inside BuildConfigurationKey.
      // Unfortunately, we can't use BuildConfigurationKey as BuildVariableNode, because it contains
      // clientEnvironment and we would have to invalidate ConfigurationCollectionNode each time
      // when any variable in client environment changes.
      BuildVariableNode.BUILD_OPTIONS.get(env);
      BuildVariableNode.TEST_ENVIRONMENT_VARIABLES.get(env);
      BuildVariableNode.BLAZE_DIRECTORIES.get(env);
      Preconditions.checkState(!env.depsMissing(),
          "BuildOptions and TestEnvironment must be created already");

      collection = configurationFactory.get().getConfigurationsInSkyframe(reporter,
          new SkyframePackageLoaderWithNodeEnvironment(env), configurationKey.get());
      // BuildConfigurationCollection can be created, but dependencies to some files might be
      // missing. In that case we need to build configurationCollection second time.
      if (env.depsMissing()) {
        return null;
      }
      // For non-incremental builds the configuration collection is not going to be cached.
      for (BuildConfiguration config : collection.getTargetConfigurations()) {
        if (!config.supportsIncrementalBuild()) {
          BuildVariableNode.BUILD_ID.get(env);
        }
      }
    } catch (InvalidConfigurationException e) {
      throw new ConfigurationCollectionNodeBuilderException(nodeKey, e);
    }
    return new ConfigurationCollectionNode(collection);
  }

  /**
   * Repeats functionality of {@link SkyframePackageLoader} but uses
   * {@link NodeBuilder.Environment#getDep} instead of {@link AutoUpdatingGraph#update}
   * for node evaluation
   */
  static class SkyframePackageLoaderWithNodeEnvironment implements LoadedPackageProvider {
    private final NodeBuilder.Environment env;

    public SkyframePackageLoaderWithNodeEnvironment(NodeBuilder.Environment env) {
      this.env = env;
    }

    private Package getPackage(String packageName) throws NoSuchPackageException{
      NodeKey key = PackageNode.key(new PathFragment(packageName));
      PackageNode node = (PackageNode) env.getDepOrThrow(key, NoSuchPackageException.class);
      Preconditions.checkNotNull(node, "Package was not loaded");
      return node.getPackage();
    }

    @Override
    public Package getLoadedPackage(final String packageName) throws NoSuchPackageException {
      try {
        return getPackage(packageName);
      } catch (NoSuchPackageException e) {
        if (e.getPackage() != null) {
          return e.getPackage();
        }
        throw e;
      }
    }

    @Override
    public Target getLoadedTarget(Label label) throws NoSuchPackageException,
        NoSuchTargetException {
      return getLoadedPackage(label.getPackageName()).getTarget(label.getName());
    }

    @Override
    public boolean isTargetCurrent(Target target) {
      throw new UnsupportedOperationException("This method is supposed not to be called");
    }

    @Override
    public void addDependency(Package pkg, String fileName) throws SyntaxException {
      Label label = Label.create(pkg.getName(), fileName);
      LabelAndConfiguration lac = new LabelAndConfiguration(label, null);
      Path pathToArtifact = pkg.getPackageDirectory().getRelative(fileName);
      Artifact artifact = new Artifact(pathToArtifact,
          Root.asSourceRoot(pkg.getSourceRoot()),
          pkg.getNameFragment().getRelative(fileName),
          lac);
      
      env.getDep(FileNode.key(artifact));
    }
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ConfigurationCollectionNodeBuilder#build}.
   */
  private static final class ConfigurationCollectionNodeBuilderException extends
      NodeBuilderException {
    public ConfigurationCollectionNodeBuilderException(NodeKey key,
        InvalidConfigurationException e) {
      super(key, e);
    }
  }
}
