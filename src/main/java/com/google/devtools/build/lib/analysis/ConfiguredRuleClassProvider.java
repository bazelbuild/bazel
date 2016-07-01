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
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
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
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.SkylarkModules;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.syntax.Environment.Phase;
import com.google.devtools.build.lib.syntax.Mutability;
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
    private final StringBuilder defaultWorkspaceFilePrefix = new StringBuilder();
    private final StringBuilder defaultWorkspaceFileSuffix = new StringBuilder();
    private Label preludeLabel;
    private String runfilesPrefix;
    private String toolsRepository;
    private final List<ConfigurationFragmentFactory> configurationFragments = new ArrayList<>();
    private final List<BuildInfoFactory> buildInfoFactories = new ArrayList<>();
    private final List<Class<? extends FragmentOptions>> configurationOptions = new ArrayList<>();

    private final Map<String, RuleClass> ruleClassMap = new HashMap<>();
    private final Map<String, RuleDefinition> ruleDefinitionMap = new HashMap<>();
    private final Map<String, NativeAspectClass> nativeAspectClassMap =
        new HashMap<>();
    private final Map<Class<? extends RuleDefinition>, RuleClass> ruleMap = new HashMap<>();
    private final Digraph<Class<? extends RuleDefinition>> dependencyGraph =
        new Digraph<>();
    private ConfigurationCollectionFactory configurationCollectionFactory;
    private Class<? extends BuildConfiguration.Fragment> universalFragment;
    private PrerequisiteValidator prerequisiteValidator;
    private ImmutableMap<String, Object> skylarkAccessibleTopLevels = ImmutableMap.of();
    private ImmutableList.Builder<Class<?>> skylarkModules =
        ImmutableList.<Class<?>>builder().addAll(SkylarkModules.MODULES);
    private ImmutableBiMap<String, Class<? extends TransitiveInfoProvider>>
        registeredSkylarkProviders = ImmutableBiMap.of();


    public void addWorkspaceFilePrefix(String contents) {
      defaultWorkspaceFilePrefix.append(contents);
    }

    public void addWorkspaceFileSuffix(String contents) {
      defaultWorkspaceFileSuffix.append(contents);
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
      ruleDefinitionMap.put(ruleDefinitionClass.getName(), ruleDefinition);
      dependencyGraph.createNode(ruleDefinitionClass);
      for (Class<? extends RuleDefinition> ancestor : ruleDefinition.getMetadata().ancestors()) {
        dependencyGraph.addEdge(ancestor, ruleDefinitionClass);
      }

      return this;
    }

    public Builder addNativeAspectClass(NativeAspectClass aspectFactoryClass) {
      nativeAspectClassMap.put(aspectFactoryClass.getName(), aspectFactoryClass);
      return this;
    }

    public Builder addConfigurationOptions(Class<? extends FragmentOptions> configurationOptions) {
      this.configurationOptions.add(configurationOptions);
      return this;
    }

    public Builder addConfigurationOptions(
        Collection<Class<? extends FragmentOptions>> optionsClasses) {
      this.configurationOptions.addAll(optionsClasses);
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

    public Builder setSkylarkAccessibleTopLevels(ImmutableMap<String, Object> objects) {
      this.skylarkAccessibleTopLevels = objects;
      return this;
    }

    public Builder addSkylarkModule(Class<?>... modules) {
      this.skylarkModules.add(modules);
      return this;
    }

    /**
     * Registers a map that indicates which keys in structs returned by skylark rules should be
     * interpreted as native TransitiveInfoProvider instances of type (map value).
     */
    public Builder setSkylarkProviderRegistry(
        ImmutableBiMap<String, Class<? extends TransitiveInfoProvider>> providers) {
      this.registeredSkylarkProviders = providers;
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
      RuleDefinition instance = checkNotNull(ruleDefinitionMap.get(definitionClass.getName()),
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
      ruleDefinitionMap.put(ruleClass.getName(), instance);

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
          ImmutableMap.copyOf(nativeAspectClassMap),
          defaultWorkspaceFilePrefix.toString(),
          defaultWorkspaceFileSuffix.toString(),
          ImmutableList.copyOf(buildInfoFactories),
          ImmutableList.copyOf(configurationOptions),
          ImmutableList.copyOf(configurationFragments),
          configurationCollectionFactory,
          universalFragment,
          prerequisiteValidator,
          skylarkAccessibleTopLevels,
          skylarkModules.build(),
          registeredSkylarkProviders);
    }

    @Override
    public Label getLabel(String labelValue) {
      return LABELS.getUnchecked(labelValue);
    }

    @Override
    public Label getToolsLabel(String labelValue) {
      return getLabel(toolsRepository + labelValue);
    }

    @Override
    public String getToolsRepository() {
      return toolsRepository;
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
   * Default content that should be added at the beginning of the WORKSPACE file.
   */
  private final String defaultWorkspaceFilePrefix;

  /**
   * Default content that should be added at the end of the WORKSPACE file.
   */
  private final String defaultWorkspaceFileSuffix;


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
   * Maps rule class name to the rule definition objects.
   */
  private final ImmutableMap<String, RuleDefinition> ruleDefinitionMap;

  /**
   * Maps aspect name to the aspect factory meta class.
   */
  private final ImmutableMap<String, NativeAspectClass> nativeAspectClassMap;

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

  private final ImmutableBiMap<String, Class<? extends TransitiveInfoProvider>>
      registeredSkylarkProviders;

  private ConfiguredRuleClassProvider(
      Label preludeLabel,
      String runfilesPrefix,
      String toolsRepository,
      ImmutableMap<String, RuleClass> ruleClassMap,
      ImmutableMap<String, RuleDefinition> ruleDefinitionMap,
      ImmutableMap<String, NativeAspectClass> nativeAspectClassMap,
      String defaultWorkspaceFilePrefix,
      String defaultWorkspaceFileSuffix,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      ImmutableList<Class<? extends FragmentOptions>> configurationOptions,
      ImmutableList<ConfigurationFragmentFactory> configurationFragments,
      ConfigurationCollectionFactory configurationCollectionFactory,
      Class<? extends BuildConfiguration.Fragment> universalFragment,
      PrerequisiteValidator prerequisiteValidator,
      ImmutableMap<String, Object> skylarkAccessibleJavaClasses,
      ImmutableList<Class<?>> skylarkModules,
      ImmutableBiMap<String, Class<? extends TransitiveInfoProvider>> registeredSkylarkProviders) {
    this.preludeLabel = preludeLabel;
    this.runfilesPrefix = runfilesPrefix;
    this.toolsRepository = toolsRepository;
    this.ruleClassMap = ruleClassMap;
    this.ruleDefinitionMap = ruleDefinitionMap;
    this.nativeAspectClassMap = nativeAspectClassMap;
    this.defaultWorkspaceFilePrefix = defaultWorkspaceFilePrefix;
    this.defaultWorkspaceFileSuffix = defaultWorkspaceFileSuffix;
    this.buildInfoFactories = buildInfoFactories;
    this.configurationOptions = configurationOptions;
    this.configurationFragments = configurationFragments;
    this.configurationCollectionFactory = configurationCollectionFactory;
    this.universalFragment = universalFragment;
    this.prerequisiteValidator = prerequisiteValidator;
    this.globals = createGlobals(skylarkAccessibleJavaClasses, skylarkModules);
    this.registeredSkylarkProviders = registeredSkylarkProviders;
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
  public Map<String, NativeAspectClass> getNativeAspectClassMap() {
    return nativeAspectClassMap;
  }

  @Override
  public NativeAspectClass getNativeAspectClass(String key) {
    return nativeAspectClassMap.get(key);
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
  public RuleDefinition getRuleClassDefinition(String ruleClassName) {
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
  public String getDefaultsPackageContent(InvocationPolicy invocationPolicy) {
    return DefaultsPackage.getDefaultsPackageContent(configurationOptions, invocationPolicy);
  }

  /**
   * Returns the defaults package for the given options taken from an optionsProvider.
   */
  public String getDefaultsPackageContent(OptionsClassProvider optionsProvider) {
    return DefaultsPackage.getDefaultsPackageContent(
        BuildOptions.of(configurationOptions, optionsProvider));
  }

  /**
   * Returns a map that indicates which keys in structs returned by skylark rules should be
   * interpreted as native TransitiveInfoProvider instances of type (map value).
   *
   * <p>That is, if this map contains "dummy" -> DummyProvider.class, a "dummy" entry in a skylark
   * rule implementations returned struct will be exported from that ConfiguredTarget as a
   * DummyProvider.
   */
  public ImmutableBiMap<String, Class<? extends TransitiveInfoProvider>>
      getRegisteredSkylarkProviders() {
    return this.registeredSkylarkProviders;
  }

  /**
   * Creates a BuildOptions class for the given options taken from an optionsProvider.
   */
  public BuildOptions createBuildOptions(OptionsClassProvider optionsProvider) {
    return BuildOptions.of(configurationOptions, optionsProvider);
  }

  private Environment.Frame createGlobals(
      ImmutableMap<String, Object> skylarkAccessibleToplLevels,
      ImmutableList<Class<?>> modules) {
    try (Mutability mutability = Mutability.create("ConfiguredRuleClassProvider globals")) {
      Environment env = createSkylarkRuleClassEnvironment(
          mutability, SkylarkModules.getGlobals(modules), null, null, null);
      for (Map.Entry<String, Object> entry : skylarkAccessibleToplLevels.entrySet()) {
        env.setup(entry.getKey(), entry.getValue());
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
    return Environment.builder(mutability)
        .setSkylark()
        .setGlobals(globals)
        .setEventHandler(eventHandler)
        .setFileContentHashCode(astFileContentHashCode)
        .setImportedExtensions(importMap)
        .setToolsRepository(toolsRepository)
        .setPhase(Phase.LOADING)
        .build();
  }

  @Override
  public Environment createSkylarkRuleClassEnvironment(
      Label extensionLabel, Mutability mutability,
      EventHandler eventHandler,
      String astFileContentHashCode,
      Map<String, Extension> importMap) {
    return createSkylarkRuleClassEnvironment(
        mutability, globals.setLabel(extensionLabel),
        eventHandler, astFileContentHashCode, importMap);
  }

  @Override
  public String getDefaultWorkspacePrefix() {
    return defaultWorkspaceFilePrefix;
  }

  @Override
  public String getDefaultWorkspaceSuffix() {
    return defaultWorkspaceFileSuffix;
  }
}
