// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.starlarkdocextract;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions.StarlarkRuleFunction;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtension;
import com.google.devtools.build.lib.bazel.bzlmod.TagClass;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule.RepositoryRuleFunction;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.skydoc.rendering.DocstringParseException;
import com.google.devtools.build.skydoc.rendering.LabelRenderer;
import com.google.devtools.build.skydoc.rendering.StarlarkFunctionInfoExtractor;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleExtensionTagClassInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.OriginKey;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderNameGroup;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Predicate;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.Structure;

/** API documentation extractor for a compiled, loaded Starlark module. */
final class ModuleInfoExtractor {
  private final Predicate<String> isWantedQualifiedName;
  private final LabelRenderer labelRenderer;

  @VisibleForTesting
  static final AttributeInfo IMPLICIT_NAME_ATTRIBUTE_INFO =
      AttributeInfo.newBuilder()
          .setName("name")
          .setType(AttributeType.NAME)
          .setMandatory(true)
          .setDocString("A unique name for this target.")
          .build();

  @VisibleForTesting
  static final ImmutableList<AttributeInfo> IMPLICIT_REPOSITORY_RULE_ATTRIBUTES =
      ImmutableList.of(
          AttributeInfo.newBuilder()
              .setName("name")
              .setType(AttributeType.NAME)
              .setMandatory(true)
              .setDocString("A unique name for this repository.")
              .build(),
          AttributeInfo.newBuilder()
              .setName("repo_mapping")
              .setType(AttributeType.STRING_DICT)
              .setDocString(
                  "In `WORKSPACE` context only: a dictionary from local repository name to global"
                      + " repository name. This allows controls over workspace dependency"
                      + " resolution for dependencies of this repository.\n\n"
                      + "For example, an entry `\"@foo\": \"@bar\"` declares that, for any time"
                      + " this repository depends on `@foo` (such as a dependency on"
                      + " `@foo//some:target`, it should actually resolve that dependency within"
                      + " globally-declared `@bar` (`@bar//some:target`).\n\n"
                      + "This attribute is _not_ supported in `MODULE.bazel` context (when invoking"
                      + " a repository rule inside a module extension's implementation function).")
              .build());

  /**
   * Constructs an instance of {@code ModuleInfoExtractor}.
   *
   * @param isWantedQualifiedName a predicate to filter the module's qualified names. A qualified
   *     name is documented if and only if (1) each component of the qualified name is public (in
   *     other words, the first character of each component of the qualified name is alphabetic) and
   *     (2) the qualified name, or one of its ancestor qualified names, satisfies the wanted
   *     predicate.
   * @param labelRenderer a string renderer for labels.
   */
  public ModuleInfoExtractor(Predicate<String> isWantedQualifiedName, LabelRenderer labelRenderer) {
    this.isWantedQualifiedName = isWantedQualifiedName;
    this.labelRenderer = labelRenderer;
  }

  /** Extracts structured documentation for the loadable symbols of a given module. */
  public ModuleInfo extractFrom(Module module) throws ExtractionException {
    ModuleInfo.Builder builder = ModuleInfo.newBuilder();
    Optional.ofNullable(module.getDocumentation()).ifPresent(builder::setModuleDocstring);
    Optional.ofNullable(BazelModuleContext.of(module))
        .map(bazelModuleContext -> labelRenderer.render(bazelModuleContext.label()))
        .ifPresent(builder::setFile);

    // We do two traversals over the module's globals: (1) find qualified names (including any
    // nesting structs) for providers loadable from this module; (2) build the documentation
    // proto, using the information from traversal 1 for provider names references by rules and
    // attributes.
    ProviderQualifiedNameCollector providerQualifiedNameCollector =
        new ProviderQualifiedNameCollector();
    providerQualifiedNameCollector.traverse(module);
    DocumentationExtractor documentationExtractor =
        new DocumentationExtractor(
            builder,
            isWantedQualifiedName,
            labelRenderer,
            providerQualifiedNameCollector.buildQualifiedNames());
    documentationExtractor.traverse(module);
    return builder.build();
  }

  private static boolean isPublicName(String name) {
    return name.length() > 0 && Character.isAlphabetic(name.charAt(0));
  }

  /** An exception indicating that the module's API documentation could not be extracted. */
  public static class ExtractionException extends Exception {
    public ExtractionException(String message) {
      super(message);
    }

    public ExtractionException(Throwable cause) {
      super(cause);
    }

    public ExtractionException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  /**
   * A stateful visitor which traverses a Starlark module's documentable globals, recursing into
   * structs.
   */
  private abstract static class GlobalsVisitor {
    public void traverse(Module module) throws ExtractionException {
      for (var entry : module.getGlobals().entrySet()) {
        String globalSymbol = entry.getKey();
        if (isPublicName(globalSymbol)) {
          maybeVisit(globalSymbol, entry.getValue(), /* shouldVisitVerifiedForAncestor= */ false);
        }
      }
    }

    /**
     * Returns whether the visitor should visit (and possibly recurse into) the value with the given
     * qualified name. Note that the visitor will not visit global names and struct fields for which
     * {@link #isPublicName} is false, regardless of {@code shouldVisit}.
     */
    protected abstract boolean shouldVisit(String qualifiedName);

    /**
     * @param qualifiedName the name under which the value may be accessed by a user of the module;
     *     for example, "foo.bar" for field bar of global struct foo
     * @param value the Starlark value
     * @param shouldVisitVerifiedForAncestor whether {@link #shouldVisit} was verified true for an
     *     ancestor struct's qualified name; e.g. {@code qualifiedName} is "a.b.c.d" and {@code
     *     shouldVisit("a.b") == true}
     */
    private void maybeVisit(
        String qualifiedName, Object value, boolean shouldVisitVerifiedForAncestor)
        throws ExtractionException {
      if (shouldVisitVerifiedForAncestor || shouldVisit(qualifiedName)) {
        if (value instanceof StarlarkRuleFunction) {
          visitRule(qualifiedName, (StarlarkRuleFunction) value);
        } else if (value instanceof StarlarkProvider) {
          visitProvider(qualifiedName, (StarlarkProvider) value);
        } else if (value instanceof StarlarkFunction) {
          visitFunction(qualifiedName, (StarlarkFunction) value);
        } else if (value instanceof StarlarkDefinedAspect) {
          visitAspect(qualifiedName, (StarlarkDefinedAspect) value);
        } else if (value instanceof RepositoryRuleFunction) {
          visitRepositoryRule(qualifiedName, (RepositoryRuleFunction) value);
        } else if (value instanceof ModuleExtension) {
          visitModuleExtension(qualifiedName, (ModuleExtension) value);
        } else if (value instanceof Structure) {
          recurseIntoStructure(
              qualifiedName, (Structure) value, /* shouldVisitVerifiedForAncestor= */ true);
        }
      } else if (value instanceof Structure) {
        recurseIntoStructure(
            qualifiedName, (Structure) value, /* shouldVisitVerifiedForAncestor= */ false);
      }
      // If the value is a constant (string, list etc.), we currently don't have a convention for
      // associating a doc string with one - so we don't emit documentation for it.
      // TODO(b/276733504): should we recurse into dicts to search for documentable values? Note
      // that dicts (unlike structs!) can have reference cycles, so we would need to track the set
      // of traversed entities.
    }

    protected void visitRule(String qualifiedName, StarlarkRuleFunction value)
        throws ExtractionException {}

    protected void visitProvider(String qualifiedName, StarlarkProvider value) {}

    protected void visitFunction(String qualifiedName, StarlarkFunction value)
        throws ExtractionException {}

    protected void visitAspect(String qualifiedName, StarlarkDefinedAspect aspect)
        throws ExtractionException {}

    protected void visitModuleExtension(String qualifiedName, ModuleExtension moduleExtension)
        throws ExtractionException {}

    protected void visitRepositoryRule(
        String qualifiedName, RepositoryRuleFunction repositoryRuleFunction)
        throws ExtractionException {}

    private void recurseIntoStructure(
        String qualifiedName, Structure structure, boolean shouldVisitVerifiedForAncestor)
        throws ExtractionException {
      for (String fieldName : structure.getFieldNames()) {
        if (isPublicName(fieldName)) {
          try {
            Object fieldValue = structure.getValue(fieldName);
            if (fieldValue != null) {
              maybeVisit(
                  String.format("%s.%s", qualifiedName, fieldName),
                  fieldValue,
                  shouldVisitVerifiedForAncestor);
            }
          } catch (EvalException e) {
            throw new ExtractionException(
                String.format(
                    "in struct %s field %s: failed to read value", qualifiedName, fieldName),
                e);
          }
        }
      }
    }
  }

  /**
   * A {@link GlobalsVisitor} which finds the qualified names (including any nesting structs) for
   * providers loadable from this module.
   */
  private static final class ProviderQualifiedNameCollector extends GlobalsVisitor {
    private final LinkedHashMap<StarlarkProvider.Key, String> qualifiedNames =
        new LinkedHashMap<>();

    /**
     * Builds a map from the keys of the Starlark providers which were walked via {@link #traverse}
     * to the qualified names (including any structs) under which those providers may be accessed by
     * a user of this module.
     *
     * <p>If the same provider is accessible under multiple names, the first documentable name wins.
     */
    public ImmutableMap<StarlarkProvider.Key, String> buildQualifiedNames() {
      return ImmutableMap.copyOf(qualifiedNames);
    }

    /**
     * Returns true always.
     *
     * <p>{@link ProviderQualifiedNameCollector} traverses all loadable providers, not filtering by
     * ModuleInfoExtractor#isWantedQualifiedName, because a non-wanted provider symbol may still be
     * referred to by a wanted rule; we do not want the provider names emitted in rule documentation
     * to vary when we change the isWantedQualifiedName filter.
     */
    @Override
    protected boolean shouldVisit(String qualifiedName) {
      return true;
    }

    @Override
    protected void visitProvider(String qualifiedName, StarlarkProvider value) {
      qualifiedNames.putIfAbsent(value.getKey(), qualifiedName);
    }
  }

  /** A {@link GlobalsVisitor} which extracts documentation for symbols in this module. */
  private static final class DocumentationExtractor extends GlobalsVisitor {
    private final ModuleInfo.Builder moduleInfoBuilder;
    private final Predicate<String> isWantedQualifiedName;
    private final LabelRenderer labelRenderer;
    private final ImmutableMap<StarlarkProvider.Key, String> providerQualifiedNames;

    /**
     * @param moduleInfoBuilder builder to which {@link #traverse} adds extracted documentation
     * @param isWantedQualifiedName a predicate to filter the module's qualified names. A qualified
     *     name is documented if and only if (1) each component of the qualified name is public (in
     *     other words, the first character of each component of the qualified name is alphabetic)
     *     and (2) the qualified name, or one of its ancestor qualified names, satisfies the wanted
     *     predicate.
     * @param labelRenderer a function for stringifying labels
     * @param providerQualifiedNames a map from the keys of documentable Starlark providers loadable
     *     from this module to the qualified names (including structure namespaces) under which
     *     those providers are accessible to a user of this module
     */
    DocumentationExtractor(
        ModuleInfo.Builder moduleInfoBuilder,
        Predicate<String> isWantedQualifiedName,
        LabelRenderer labelRenderer,
        ImmutableMap<StarlarkProvider.Key, String> providerQualifiedNames) {
      this.moduleInfoBuilder = moduleInfoBuilder;
      this.isWantedQualifiedName = isWantedQualifiedName;
      this.labelRenderer = labelRenderer;
      this.providerQualifiedNames = providerQualifiedNames;
    }

    @Override
    protected boolean shouldVisit(String qualifiedName) {
      return isWantedQualifiedName.test(qualifiedName);
    }

    @Override
    protected void visitFunction(String qualifiedName, StarlarkFunction function)
        throws ExtractionException {
      try {
        moduleInfoBuilder.addFuncInfo(
            StarlarkFunctionInfoExtractor.fromNameAndFunction(
                qualifiedName, function, /* withOriginKey= */ true, labelRenderer));
      } catch (DocstringParseException e) {
        throw new ExtractionException(e);
      }
    }

    @Override
    protected void visitRule(String qualifiedName, StarlarkRuleFunction ruleFunction)
        throws ExtractionException {
      RuleInfo.Builder ruleInfoBuilder = RuleInfo.newBuilder();
      // Record the name under which this symbol is made accessible, which may differ from the
      // symbol's exported name
      ruleInfoBuilder.setRuleName(qualifiedName);
      // ... but record the origin rule key for cross references.
      ruleInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              .setName(ruleFunction.getName())
              .setFile(labelRenderer.render(ruleFunction.getExtensionLabel())));
      ruleFunction.getDocumentation().ifPresent(ruleInfoBuilder::setDocString);

      RuleClass ruleClass = ruleFunction.getRuleClass();
      if (ruleClass.getRuleClassType() == RuleClassType.TEST) {
        ruleInfoBuilder.setTest(true);
      }
      if (ruleClass.hasAttr("$is_executable", Type.BOOLEAN)) {
        ruleInfoBuilder.setExecutable(true);
      }

      ruleInfoBuilder.addAttribute(IMPLICIT_NAME_ATTRIBUTE_INFO); // name comes first
      addDocumentableAttributes(
          ruleClass.getAttributes(), ruleInfoBuilder::addAttribute, "rule " + qualifiedName);
      ImmutableSet<StarlarkProviderIdentifier> advertisedProviders =
          ruleClass.getAdvertisedProviders().getStarlarkProviders();
      if (!advertisedProviders.isEmpty()) {
        ruleInfoBuilder.setAdvertisedProviders(buildProviderNameGroup(advertisedProviders));
      }
      moduleInfoBuilder.addRuleInfo(ruleInfoBuilder);
    }

    @Override
    protected void visitProvider(String qualifiedName, StarlarkProvider provider) {
      ProviderInfo.Builder providerInfoBuilder = ProviderInfo.newBuilder();
      // Record the name under which this symbol is made accessible, which may differ from the
      // symbol's exported name.
      // Note that it's possible that qualifiedName != getDocumentedProviderName() if the same
      // provider symbol is made accessible under more than one qualified name.
      // TODO(b/276733504): if a provider (or any other documentable entity) is made accessible
      // under two different public qualified names, record them in a repeated field inside a single
      // ProviderInfo (or other ${FOO}Info for documentable entity ${FOO}) message, instead of
      // producing a separate ${FOO}Info message for each alias. That requires adding an "alias"
      // field to ${FOO}Info messages (making the existing "${FOO}_name" field repeated would break
      // existing Stardoc templates). Note that for backwards compatibility,
      // ProviderNameGroup.provider_name would still need to refer to only the first qualified name
      // under which a given provider is made accessible by the module.
      providerInfoBuilder.setProviderName(qualifiedName);
      // Record the origin provider key for cross references.
      providerInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              .setName(provider.getName())
              .setFile(labelRenderer.render(provider.getKey().getExtensionLabel())));
      provider.getDocumentation().ifPresent(providerInfoBuilder::setDocString);
      ImmutableMap<String, Optional<String>> schema = provider.getSchema();
      if (schema != null) {
        for (Map.Entry<String, Optional<String>> entry : schema.entrySet()) {
          if (isPublicName(entry.getKey())) {
            ProviderFieldInfo.Builder fieldInfoBuilder = ProviderFieldInfo.newBuilder();
            fieldInfoBuilder.setName(entry.getKey());
            entry.getValue().ifPresent(fieldInfoBuilder::setDocString);
            providerInfoBuilder.addFieldInfo(fieldInfoBuilder.build());
          }
        }
      }
      moduleInfoBuilder.addProviderInfo(providerInfoBuilder);
    }

    @Override
    protected void visitAspect(String qualifiedName, StarlarkDefinedAspect aspect)
        throws ExtractionException {
      AspectInfo.Builder aspectInfoBuilder = AspectInfo.newBuilder();
      // Record the name under which this symbol is made accessible, which may differ from the
      // symbol's exported name
      aspectInfoBuilder.setAspectName(qualifiedName);
      // ... but record the origin aspect key for cross references.
      aspectInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              .setName(aspect.getAspectClass().getExportedName())
              .setFile(labelRenderer.render(aspect.getAspectClass().getExtensionLabel())));
      aspect.getDocumentation().ifPresent(aspectInfoBuilder::setDocString);
      for (String aspectAttribute : aspect.getAttributeAspects()) {
        if (isPublicName(aspectAttribute)) {
          aspectInfoBuilder.addAspectAttribute(aspectAttribute);
        }
      }
      aspectInfoBuilder.addAttribute(IMPLICIT_NAME_ATTRIBUTE_INFO); // name comes first
      addDocumentableAttributes(
          aspect.getAttributes(), aspectInfoBuilder::addAttribute, "aspect " + qualifiedName);
      moduleInfoBuilder.addAspectInfo(aspectInfoBuilder);
    }

    @Override
    protected void visitModuleExtension(String qualifiedName, ModuleExtension moduleExtension)
        throws ExtractionException {
      ModuleExtensionInfo.Builder moduleExtensionInfoBuilder = ModuleExtensionInfo.newBuilder();
      moduleExtensionInfoBuilder.setExtensionName(qualifiedName);
      moduleExtensionInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              // TODO(arostovtsev): attempt to retrieve the name under which the module was
              // originally defined so we can call setName() too. The easiest solution might be to
              // make ModuleExtension a StarlarkExportable (partially reverting cl/513213080).
              // Alternatively, we'd need to search the defining module's globals, similarly to what
              // we do in FunctionUtil#getFunctionOriginKey.
              .setFile(labelRenderer.render(moduleExtension.getDefiningBzlFileLabel())));
      moduleExtension.getDoc().ifPresent(moduleExtensionInfoBuilder::setDocString);
      for (Map.Entry<String, TagClass> entry : moduleExtension.getTagClasses().entrySet()) {
        ModuleExtensionTagClassInfo.Builder tagClassInfoBuilder =
            ModuleExtensionTagClassInfo.newBuilder();
        tagClassInfoBuilder.setTagName(entry.getKey());
        entry.getValue().getDoc().ifPresent(tagClassInfoBuilder::setDocString);
        addDocumentableAttributes(
            entry.getValue().getAttributes(),
            tagClassInfoBuilder::addAttribute,
            String.format("module extension %s tag class %s", qualifiedName, entry.getKey()));
        moduleExtensionInfoBuilder.addTagClass(tagClassInfoBuilder);
      }
      moduleInfoBuilder.addModuleExtensionInfo(moduleExtensionInfoBuilder);
    }

    @Override
    protected void visitRepositoryRule(
        String qualifiedName, RepositoryRuleFunction repositoryRuleFunction)
        throws ExtractionException {
      RepositoryRuleInfo.Builder repositoryRuleInfoBuilder = RepositoryRuleInfo.newBuilder();
      repositoryRuleInfoBuilder.setRuleName(qualifiedName);
      repositoryRuleFunction.getDocumentation().ifPresent(repositoryRuleInfoBuilder::setDocString);
      RuleClass ruleClass = repositoryRuleFunction.getRuleClass();
      repositoryRuleInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              .setName(ruleClass.getName())
              .setFile(labelRenderer.render(repositoryRuleFunction.getExtensionLabel())));

      repositoryRuleInfoBuilder.addAllAttribute(IMPLICIT_REPOSITORY_RULE_ATTRIBUTES);
      addDocumentableAttributes(
          ruleClass.getAttributes(),
          repositoryRuleInfoBuilder::addAttribute,
          "repository rule " + qualifiedName);
      if (ruleClass.hasAttr("$environ", Type.STRING_LIST)) {
        repositoryRuleInfoBuilder.addAllEnviron(
            Type.STRING_LIST.cast(ruleClass.getAttributeByName("$environ").getDefaultValue(null)));
      }
      moduleInfoBuilder.addRepositoryRuleInfo(repositoryRuleInfoBuilder);
    }

    private static AttributeType getAttributeType(Attribute attribute, String where)
        throws ExtractionException {
      Type<?> type = attribute.getType();
      if (type.equals(Type.INTEGER)) {
        return AttributeType.INT;
      } else if (type.equals(BuildType.LABEL)) {
        return AttributeType.LABEL;
      } else if (type.equals(Type.STRING)) {
        if (attribute.getPublicName().equals("name")) {
          return AttributeType.NAME;
        } else {
          return AttributeType.STRING;
        }
      } else if (type.equals(Type.STRING_LIST)) {
        return AttributeType.STRING_LIST;
      } else if (type.equals(Type.INTEGER_LIST)) {
        return AttributeType.INT_LIST;
      } else if (type.equals(BuildType.LABEL_LIST)) {
        return AttributeType.LABEL_LIST;
      } else if (type.equals(Type.BOOLEAN)) {
        return AttributeType.BOOLEAN;
      } else if (type.equals(BuildType.LABEL_KEYED_STRING_DICT)) {
        return AttributeType.LABEL_STRING_DICT;
      } else if (type.equals(Type.STRING_DICT)) {
        return AttributeType.STRING_DICT;
      } else if (type.equals(Type.STRING_LIST_DICT)) {
        return AttributeType.STRING_LIST_DICT;
      } else if (type.equals(BuildType.OUTPUT)) {
        return AttributeType.OUTPUT;
      } else if (type.equals(BuildType.OUTPUT_LIST)) {
        return AttributeType.OUTPUT_LIST;
      } else if (type.equals(BuildType.LICENSE)) {
        // TODO(https://github.com/bazelbuild/bazel/issues/6420): deprecated, disabled in Bazel by
        // default, broken and with almost no remaining users, so we don't have an AttributeType for
        // it. Until this type is removed, following the example of legacy Stardoc, pretend it's a
        // list of strings.
        return AttributeType.STRING_LIST;
      }

      throw new ExtractionException(
          String.format(
              "in %s attribute %s: unsupported type %s",
              where, attribute.getPublicName(), type.getClass().getSimpleName()));
    }

    private AttributeInfo buildAttributeInfo(Attribute attribute, String where)
        throws ExtractionException {
      AttributeInfo.Builder builder = AttributeInfo.newBuilder();
      builder.setName(attribute.getPublicName());
      Optional.ofNullable(attribute.getDoc()).ifPresent(builder::setDocString);
      builder.setType(getAttributeType(attribute, where));
      builder.setMandatory(attribute.isMandatory());
      if (!attribute.isConfigurable()) {
        builder.setNonconfigurable(true);
      }
      for (ImmutableSet<StarlarkProviderIdentifier> providerGroup :
          attribute.getRequiredProviders().getStarlarkProviders()) {
        // TODO(b/290788853): it is meaningless to require a provider on an attribute of a
        // repository rule or of a module extension tag.
        builder.addProviderNameGroup(buildProviderNameGroup(providerGroup));
      }

      if (!attribute.isMandatory()) {
        Object defaultValue = Attribute.valueToStarlark(attribute.getDefaultValueUnchecked());
        builder.setDefaultValue(labelRenderer.reprWithoutLabelConstructor(defaultValue));
      }
      return builder.build();
    }

    private void addDocumentableAttributes(
        Iterable<Attribute> attributes, Consumer<AttributeInfo> builder, String where)
        throws ExtractionException {
      for (Attribute attribute : attributes) {
        if (attribute.starlarkDefined()
            && attribute.isDocumented()
            && isPublicName(attribute.getPublicName())) {
          builder.accept(buildAttributeInfo(attribute, where));
        }
      }
    }

    /**
     * Returns the provider name suitable for use in this module's documentation. For a provider
     * loadable from this module, this is the qualified name (or more precisely, the first qualified
     * name) under which a user of this module may access it. For local providers and for providers
     * loaded but not re-exported via a global, it's the provider key name (a.k.a. {@code
     * provider.toString()}). For legacy struct providers, it's the legacy ID (which also happens to
     * be {@code provider.toString()}).
     */
    private String getDocumentedProviderName(StarlarkProviderIdentifier provider) {
      if (!provider.isLegacy()) {
        String qualifiedName = providerQualifiedNames.get(provider.getKey());
        if (qualifiedName != null) {
          return qualifiedName;
        }
      }
      return provider.toString();
    }

    private ProviderNameGroup buildProviderNameGroup(
        ImmutableSet<StarlarkProviderIdentifier> providerGroup) {
      ProviderNameGroup.Builder providerNameGroupBuilder = ProviderNameGroup.newBuilder();
      for (StarlarkProviderIdentifier provider : providerGroup) {
        providerNameGroupBuilder.addProviderName(getDocumentedProviderName(provider));
        OriginKey.Builder providerKeyBuilder = OriginKey.newBuilder().setName(provider.toString());
        if (!provider.isLegacy()) {
          if (provider.getKey() instanceof StarlarkProvider.Key) {
            Label definingModule = ((StarlarkProvider.Key) provider.getKey()).getExtensionLabel();
            providerKeyBuilder.setFile(labelRenderer.render(definingModule));
          } else if (provider.getKey() instanceof BuiltinProvider.Key) {
            providerKeyBuilder.setFile("<native>");
          }
        }
        providerNameGroupBuilder.addOriginKey(providerKeyBuilder.build());
      }
      return providerNameGroupBuilder.build();
    }
  }
}
