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
package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType.ABSTRACT;
import static com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType.TEST;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.DefaultsPackage;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NativeAspectClass.NativeAspectFactory;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.SkylarkModules;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.common.options.OptionsClassProvider;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Knows about every rule Blaze supports and the associated configuration options.
 *
 * <p>This class is initialized on server startup and the set of rules, build info factories
 * and configuration options is guarantees not to change over the life time of the Blaze server.
 */
public class ConfiguredRuleClassProvider implements RuleClassProvider {

  /**
   * Custom dependency validation logic.
   */
  public interface PrerequisiteValidator {
    /**
     * Checks whether the rule in {@code contextBuilder} is allowed to depend on
     * {@code prerequisite} through the attribute {@code attribute}.
     *
     * <p>Can be used for enforcing any organization-specific policies about the layout of the
     * workspace.
     */
    void validate(
        RuleContext.Builder contextBuilder, ConfiguredTarget prerequisite, Attribute attribute);
  }

  /**
   * Builder for {@link ConfiguredRuleClassProvider}.
   */
  public static class Builder implements RuleDefinitionEnvironment {
    private final StringBuilder defaultWorkspaceFile = new StringBuilder();
    private Label preludeLabel;
    private String runfilesPrefix;
    private String toolsRepository;
    private final List<ConfigurationFragmentFactory> configurationFragments = new ArrayList<>();
    private final List<BuildInfoFactory> buildInfoFactories = new ArrayList<>();
    private final List<Class<? extends FragmentOptions>> configurationOptions = new ArrayList<>();

    private final Map<String, RuleClass> ruleClassMap = new HashMap<>();
    private final  Map<String, Class<? extends RuleDefinition>> ruleDefinitionMap =
        new HashMap<>();
    private final Map<String, Class<? extends NativeAspectFactory>> aspectFactoryMap =
        new HashMap<>();
    private final Map<Class<? extends RuleDefinition>, RuleClass> ruleMap = new HashMap<>();
    private final Map<Class<? extends RuleDefinition>, RuleDefinition> ruleDefinitionInstanceCache =
        new HashMap<>();
    private final Digraph<Class<? extends RuleDefinition>> dependencyGraph =
        new Digraph<>();
    private ConfigurationCollectionFactory configurationCollectionFactory;
    private Class<? extends BuildConfiguration.Fragment> universalFragment;
    private PrerequisiteValidator prerequisiteValidator;
    private ImmutableMap<String, SkylarkType> skylarkAccessibleJavaClasses = ImmutableMap.of();
    private ImmutableList.Builder<Class<?>> skylarkModules =
        ImmutableList.<Class<?>>builder().addAll(SkylarkModules.MODULES);
    private final List<Class<? extends FragmentOptions>> buildOptions = Lists.newArrayList();

    public void addWorkspaceFile(String contents) {
      defaultWorkspaceFile.append(contents);
    }

    public Builder setPrelude(String preludeLabelString) {
      try {
        this.preludeLabel = Label.parseAbsolute(preludeLabelString);
      } catch (LabelSyntaxException e) {
        String errorMsg =
            String.format("Prelude label '%s' is invalid: %s", preludeLabelString, e.getMessage());
        throw new IllegalArgumentException(errorMsg);
      }
      return this;
    }

    public Builder addBuildOptions(Collection<Class<? extends FragmentOptions>> optionsClasses) {
      buildOptions.addAll(optionsClasses);
      return this;
    }

    public Builder setRunfilesPrefix(String runfilesPrefix) {
      this.runfilesPrefix = runfilesPrefix;
      return this;
    }

    public Builder setToolsRepository(String toolsRepository) {
      this.toolsRepository = toolsRepository;
      return this;
    }

    public Builder setPrerequisiteValidator(PrerequisiteValidator prerequisiteValidator) {
      this.prerequisiteValidator = prerequisiteValidator;
      return this;
    }

    public Builder addBuildInfoFactory(BuildInfoFactory factory) {
      buildInfoFactories.add(factory);
      return this;
    }

    public Builder addRuleDefinition(RuleDefinition ruleDefinition) {
      Class<? extends RuleDefinition> ruleDefinitionClass = ruleDefinition.getClass();
      ruleDefinitionInstanceCache.put(ruleDefinitionClass, ruleDefinition);
      dependencyGraph.createNode(ruleDefinitionClass);
      for (Class<? extends RuleDefinition> ancestor : ruleDefinition.getMetadata().ancestors()) {
        dependencyGraph.addEdge(ancestor, ruleDefinitionClass);
      }

      return this;
    }

    public Builder addAspectFactory(
        String name, Class<? extends ConfiguredNativeAspectFactory> configuredAspectFactoryClass) {
      aspectFactoryMap.put(name, configuredAspectFactoryClass);

      return this;
    }

    public Builder addConfigurationOptions(Class<? extends FragmentOptions> configurationOptions) {
      this.configurationOptions.add(configurationOptions);
      return this;
    }

    public Builder addConfigurationFragment(ConfigurationFragmentFactory factory) {
      configurationFragments.add(factory);
      return this;
    }

    public Builder setConfigurationCollectionFactory(ConfigurationCollectionFactory factory) {
      this.configurationCollectionFactory = factory;
      return this;
    }

    public Builder setUniversalConfigurationFragment(
        Class<? extends BuildConfiguration.Fragment> fragment) {
      this.universalFragment = fragment;
      return this;
    }

    public Builder setSkylarkAccessibleJavaClasses(ImmutableMap<String, SkylarkType> objects) {
      this.skylarkAccessibleJavaClasses = objects;
      return this;
    }

    public Builder addSkylarkModule(Class<?>... modules) {
      this.skylarkModules.add(modules);
      return this;
    }

    private RuleConfiguredTargetFactory createFactory(
        Class<? extends RuleConfiguredTargetFactory> factoryClass) {
      try {
        Constructor<? extends RuleConfiguredTargetFactory> ctor = factoryClass.getConstructor();
        return ctor.newInstance();
      } catch (NoSuchMethodException | IllegalAccessException | InstantiationException
          | InvocationTargetException e) {
        throw new IllegalStateException(e);
      }
    }

    private RuleClass commitRuleDefinition(Class<? extends RuleDefinition> definitionClass) {
      RuleDefinition instance = checkNotNull(ruleDefinitionInstanceCache.get(definitionClass),
          "addRuleDefinition(new %s()) should be called before build()", definitionClass.getName());

      RuleDefinition.Metadata metadata = instance.getMetadata();
      checkArgument(ruleClassMap.get(metadata.name()) == null, metadata.name());

      List<Class<? extends RuleDefinition>> ancestors = metadata.ancestors();

      checkArgument(
          metadata.type() == ABSTRACT ^ metadata.factoryClass()
              != RuleConfiguredTargetFactory.class);
      checkArgument(
          (metadata.type() != TEST)
          || ancestors.contains(BaseRuleClasses.TestBaseRule.class));

      RuleClass[] ancestorClasses = new RuleClass[ancestors.size()];
      for (int i = 0; i < ancestorClasses.length; i++) {
        ancestorClasses[i] = ruleMap.get(ancestors.get(i));
        if (ancestorClasses[i] == null) {
          // Ancestors should have been initialized by now
          throw new IllegalStateException("Ancestor " + ancestors.get(i) + " of "
              + metadata.name() + " is not initialized");
        }
      }

      RuleConfiguredTargetFactory factory = null;
      if (metadata.type() != ABSTRACT) {
        factory = createFactory(metadata.factoryClass());
      }

      RuleClass.Builder builder = new RuleClass.Builder(
          metadata.name(), metadata.type(), false, ancestorClasses);
      builder.factory(factory);
      RuleClass ruleClass = instance.build(builder, this);
      ruleMap.put(definitionClass, ruleClass);
      ruleClassMap.put(ruleClass.getName(), ruleClass);
      ruleDefinitionMap.put(ruleClass.getName(), definitionClass);

      return ruleClass;
    }

    public ConfiguredRuleClassProvider build() {
      for (Node<Class<? extends RuleDefinition>> ruleDefinition :
          dependencyGraph.getTopologicalOrder()) {
        commitRuleDefinition(ruleDefinition.getLabel());
      }

      return new ConfiguredRuleClassProvider(
          preludeLabel,
          runfilesPrefix,
          toolsRepository,
          ImmutableMap.copyOf(ruleClassMap),
          ImmutableMap.copyOf(ruleDefinitionMap),
          ImmutableMap.copyOf(aspectFactoryMap),
          defaultWorkspaceFile.toString(),
          ImmutableList.copyOf(buildInfoFactories),
          ImmutableList.copyOf(configurationOptions),
          ImmutableList.copyOf(configurationFragments),
          configurationCollectionFactory,
          universalFragment,
          prerequisiteValidator,
          skylarkAccessibleJavaClasses,
          skylarkModules.build(),
          buildOptions);
    }

    @Override
    public Label getLabel(String labelValue) {
      return LABELS.getUnchecked(labelValue);
    }

    @Override
    public Label getToolsLabel(String labelValue) {
      return getLabel(toolsRepository + labelValue);
    }
  }

  /**
   * Used to make the label instances unique, so that we don't create a new
   * instance for every rule.
   */
  private static final LoadingCache<String, Label> LABELS = CacheBuilder.newBuilder().build(
      new CacheLoader<String, Label>() {
    @Override
    public Label load(String from) {
      try {
        return Label.parseAbsolute(from);
      } catch (LabelSyntaxException e) {
        throw new IllegalArgumentException(from, e);
      }
    }
  });

  /**
   * A list of relative paths to the WORKSPACE files needed to provide external dependencies for
   * the rule classes.
   */
  String defaultWorkspaceFile;

  /**
   * Label for the prelude file.
   */
  private final Label preludeLabel;

  /**
   * The default runfiles prefix.
   */
  private final String runfilesPrefix;

  /**
   * The path to the tools repository.
   */
  private final String toolsRepository;

  /**
   * Maps rule class name to the metaclass instance for that rule.
   */
  private final ImmutableMap<String, RuleClass> ruleClassMap;

  /**
   * Maps rule class name to the rule definition metaclasses.
   */
  private final ImmutableMap<String, Class<? extends RuleDefinition>> ruleDefinitionMap;

  /**
   * Maps aspect name to the aspect factory meta class.
   */
  private final ImmutableMap<String, Class<? extends NativeAspectFactory>> aspectFactoryMap;

  /**
   * The configuration options that affect the behavior of the rules.
   */
  private final ImmutableList<Class<? extends FragmentOptions>> configurationOptions;

  /**
   * The set of configuration fragment factories.
   */
  private final ImmutableList<ConfigurationFragmentFactory> configurationFragments;

  /**
   * The factory that creates the configuration collection.
   */
  private final ConfigurationCollectionFactory configurationCollectionFactory;

  /**
   * A configuration fragment that should be available to all rules even when they don't
   * explicitly require it.
   */
  private final Class<? extends BuildConfiguration.Fragment> universalFragment;

  private final ImmutableList<BuildInfoFactory> buildInfoFactories;

  private final PrerequisiteValidator prerequisiteValidator;

  private final Environment.Frame globals;

  private final List<Class<? extends FragmentOptions>> buildOptions;

  private ConfiguredRuleClassProvider(
      Label preludeLabel,
      String runfilesPrefix,
      String toolsRepository,
      ImmutableMap<String, RuleClass> ruleClassMap,
      ImmutableMap<String, Class<? extends RuleDefinition>> ruleDefinitionMap,
      ImmutableMap<String, Class<? extends NativeAspectFactory>> aspectFactoryMap,
      String defaultWorkspaceFile,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      ImmutableList<Class<? extends FragmentOptions>> configurationOptions,
      ImmutableList<ConfigurationFragmentFactory> configurationFragments,
      ConfigurationCollectionFactory configurationCollectionFactory,
      Class<? extends BuildConfiguration.Fragment> universalFragment,
      PrerequisiteValidator prerequisiteValidator,
      ImmutableMap<String, SkylarkType> skylarkAccessibleJavaClasses,
      ImmutableList<Class<?>> skylarkModules,
      List<Class<? extends FragmentOptions>> buildOptions) {
    this.preludeLabel = preludeLabel;
    this.runfilesPrefix = runfilesPrefix;
    this.toolsRepository = toolsRepository;
    this.ruleClassMap = ruleClassMap;
    this.ruleDefinitionMap = ruleDefinitionMap;
    this.aspectFactoryMap = aspectFactoryMap;
    this.defaultWorkspaceFile = defaultWorkspaceFile;
    this.buildInfoFactories = buildInfoFactories;
    this.configurationOptions = configurationOptions;
    this.configurationFragments = configurationFragments;
    this.configurationCollectionFactory = configurationCollectionFactory;
    this.universalFragment = universalFragment;
    this.prerequisiteValidator = prerequisiteValidator;
    this.globals = createGlobals(skylarkAccessibleJavaClasses, skylarkModules);
    this.buildOptions = buildOptions;
  }

  public PrerequisiteValidator getPrerequisiteValidator() {
    return prerequisiteValidator;
  }

  @Override
  public Label getPreludeLabel() {
    return preludeLabel;
  }

  @Override
  public String getRunfilesPrefix() {
    return runfilesPrefix;
  }

  @Override
  public String getToolsRepository() {
    return toolsRepository;
  }

  @Override
  public Map<String, RuleClass> getRuleClassMap() {
    return ruleClassMap;
  }

  @Override
  public Map<String, Class<? extends NativeAspectFactory>> getAspectFactoryMap() {
    return aspectFactoryMap;
  }

  /**
   * Returns a list of build info factories that are needed for the supported languages.
   */
  public ImmutableList<BuildInfoFactory> getBuildInfoFactories() {
    return buildInfoFactories;
  }

  /**
   * Returns the set of configuration fragments provided by this module.
   */
  public ImmutableList<ConfigurationFragmentFactory> getConfigurationFragments() {
    return configurationFragments;
  }

  /**
   * Returns the set of configuration options that are supported in this module.
   */
  public ImmutableList<Class<? extends FragmentOptions>> getConfigurationOptions() {
    return configurationOptions;
  }

  /**
   * Returns the definition of the rule class definition with the specified name.
   */
  public Class<? extends RuleDefinition> getRuleClassDefinition(String ruleClassName) {
    return ruleDefinitionMap.get(ruleClassName);
  }

  /**
   * Returns the configuration collection creator.
   */
  public ConfigurationCollectionFactory getConfigurationCollectionFactory() {
    return configurationCollectionFactory;
  }

  /**
   * Returns the configuration fragment that should be available to all rules even when they
   * don't explicitly require it.
   */
  public Class<? extends BuildConfiguration.Fragment> getUniversalFragment() {
    return universalFragment;
  }

  /**
   * Returns the defaults package for the default settings.
   */
  public String getDefaultsPackageContent() {
    return DefaultsPackage.getDefaultsPackageContent(configurationOptions);
  }

  /**
   * Returns the defaults package for the given options taken from an optionsProvider.
   */
  public String getDefaultsPackageContent(OptionsClassProvider optionsProvider) {
    return DefaultsPackage.getDefaultsPackageContent(
        BuildOptions.of(configurationOptions, optionsProvider));
  }

  public ImmutableList<Class<? extends FragmentOptions>> getOptionFragments() {
    return ImmutableList.copyOf(buildOptions);
  }

  /**
   * Creates a BuildOptions class for the given options taken from an optionsProvider.
   */
  public BuildOptions createBuildOptions(OptionsClassProvider optionsProvider) {
    return BuildOptions.of(configurationOptions, optionsProvider);
  }

  private Environment.Frame createGlobals(
      ImmutableMap<String, SkylarkType> skylarkAccessibleJavaClasses,
      ImmutableList<Class<?>> modules) {
    try (Mutability mutability = Mutability.create("ConfiguredRuleClassProvider globals")) {
      Environment env = createSkylarkRuleClassEnvironment(
          mutability, SkylarkModules.getGlobals(modules), null, null, null);
      for (Map.Entry<String, SkylarkType> entry : skylarkAccessibleJavaClasses.entrySet()) {
        env.setup(entry.getKey(), entry.getValue().getType());
      }
      return env.getGlobals();
    }
  }

  private Environment createSkylarkRuleClassEnvironment(
      Mutability mutability,
      Environment.Frame globals,
      EventHandler eventHandler,
      String astFileContentHashCode,
      Map<String, Extension> importMap) {
    Environment env = Environment.builder(mutability)
        .setSkylark()
        .setGlobals(globals)
        .setEventHandler(eventHandler)
        .setFileContentHashCode(astFileContentHashCode)
        .setImportedExtensions(importMap)
        .setLoadingPhase()
        .build();
    return env;
  }

  @Override
  public Environment createSkylarkRuleClassEnvironment(
      Mutability mutability,
      EventHandler eventHandler,
      String astFileContentHashCode,
      Map<String, Extension> importMap) {
    return createSkylarkRuleClassEnvironment(
        mutability, globals, eventHandler, astFileContentHashCode, importMap);
  }


  @Override
  public String getDefaultWorkspaceFile() {
    return defaultWorkspaceFile;
  }
}
