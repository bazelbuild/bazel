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

package com.google.devtools.build.lib.starlarkdocextract;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions.MacroFunction;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions.StarlarkRuleFunction;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtension;
import com.google.devtools.build.lib.bazel.bzlmod.TagClass;
import com.google.devtools.build.lib.bazel.repository.RepoRule;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule.StarlarkRepoRule;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.packages.AspectPropagationEdgesSupplier.FixedListSupplier;
import com.google.devtools.build.lib.packages.AspectPropagationEdgesSupplier.FunctionSupplier;
import com.google.devtools.build.lib.packages.MacroClass;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeType;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionTagClassInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.OriginKey;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkOtherSymbolInfo;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.Structure;
import net.starlark.java.syntax.Comment;
import net.starlark.java.syntax.DocComments;
import net.starlark.java.syntax.Program;

/** API documentation extractor for a compiled, loaded Starlark module. */
public final class ModuleInfoExtractor {
  private final Predicate<String> isWantedQualifiedName;
  private final LabelRenderer labelRenderer;
  private boolean allowUnusedDocComments = false;

  @VisibleForTesting
  public static final ImmutableMap<String, AttributeInfo> IMPLICIT_MACRO_ATTRIBUTES =
      ImmutableMap.of(
          "name",
          AttributeInfo.newBuilder()
              .setName("name")
              .setType(AttributeType.NAME)
              .setMandatory(true)
              .setDocString(
                  "A unique name for this macro instance. Normally, this is also the name for the"
                      + " macro's main or only target. The names of any other targets that this"
                      + " macro might create will be this name with a string suffix.")
              .build(),
          "visibility",
          AttributeInfo.newBuilder()
              .setName("visibility")
              .setType(AttributeType.LABEL_LIST)
              .setMandatory(false)
              .setNonconfigurable(true)
              .setNativelyDefined(true)
              .setDocString(
                  "The visibility to be passed to this macro's exported targets. It always"
                      + " implicitly includes the location where this macro is instantiated, so"
                      + " this attribute only needs to be explicitly set if you want the macro's"
                      + " targets to be additionally visible somewhere else.")
              .build());

  @VisibleForTesting
  public static final ImmutableMap<String, AttributeInfo> IMPLICIT_REPOSITORY_RULE_ATTRIBUTES =
      ImmutableMap.of(
          "name",
          AttributeInfo.newBuilder()
              .setName("name")
              .setType(AttributeType.NAME)
              .setMandatory(true)
              .setDocString("A unique name for this repository.")
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

  /** Allows unused doc comments in modules. */
  @CanIgnoreReturnValue
  public ModuleInfoExtractor allowUnusedDocComments() {
    this.allowUnusedDocComments = true;
    return this;
  }

  /** Extracts structured documentation for the loadable symbols of a given module. */
  public ModuleInfo extractFrom(
      Module module,
      ImmutableMap<String, DocComments> docCommentsMap,
      ImmutableList<Comment> unusedDocCommentLines)
      throws ExtractionException {
    if (!allowUnusedDocComments && !unusedDocCommentLines.isEmpty()) {
      throw new ExtractionException(
          module,
          String.format(
              "unexpected or conflicting doc comments on line%s %s; a doc comment must be attached"
                  + " to the declaration of a global variable",
              unusedDocCommentLines.size() > 1 ? "s" : "",
              Joiner.on(", ")
                  .join(
                      unusedDocCommentLines.stream()
                          .map(c -> c.getStartLocation().line())
                          .iterator())));
    }
    ModuleInfo.Builder builder = ModuleInfo.newBuilder();
    Optional.ofNullable(module.getDocumentation())
        .map(StringEncoding::internalToUnicode)
        .ifPresent(builder::setModuleDocstring);
    Optional.ofNullable(BazelModuleContext.of(module))
        .map(
            bazelModuleContext ->
                internalToUnicode(labelRenderer.render(bazelModuleContext.label())))
        .ifPresent(builder::setFile);

    // We do two traversals over the module's globals: (1) find qualified names (including any
    // nesting structs) for providers loadable from this module; (2) build the documentation
    // proto, using the information from traversal 1 for provider names references by rules and
    // attributes.
    ProviderQualifiedNameCollector providerQualifiedNameCollector =
        new ProviderQualifiedNameCollector(module);
    providerQualifiedNameCollector.traverse();
    DocumentationExtractor documentationExtractor =
        new DocumentationExtractor(
            module,
            docCommentsMap,
            builder,
            isWantedQualifiedName,
            ExtractorContext.builder()
                .labelRenderer(labelRenderer)
                .providerQualifiedNames(providerQualifiedNameCollector.buildQualifiedNames())
                .build(),
            allowUnusedDocComments);
    documentationExtractor.traverse();
    return builder.build();
  }

  public ModuleInfo extractFrom(Module module, Program program) throws ExtractionException {
    return extractFrom(module, program.getDocCommentsMap(), program.getUnusedDocCommentLines());
  }

  public ModuleInfo extractFrom(Module module) throws ExtractionException {
    BazelModuleContext moduleContext =
        checkNotNull(
            BazelModuleContext.of(module), "Module %s does not have a BazelModuleContext", module);
    return extractFrom(
        module, moduleContext.getDocCommentsMap(), moduleContext.getUnusedDocCommentLines());
  }

  /**
   * A stateful visitor which traverses a Starlark module's documentable globals, recursing into
   * structs.
   */
  private abstract static class GlobalsVisitor {
    void traverse() throws ExtractionException {
      for (var entry : getModule().getGlobals().entrySet()) {
        String globalSymbol = entry.getKey();
        if (ExtractorContext.isPublicName(globalSymbol)) {
          maybeVisit(globalSymbol, entry.getValue(), /* shouldVisitVerifiedForAncestor= */ false);
        }
      }
    }

    abstract Module getModule();

    /**
     * Returns whether the visitor should visit (and possibly recurse into) the value with the given
     * qualified name. Note that the visitor will not visit global names and struct fields for which
     * {@link #isPublicName} is false, regardless of {@code shouldVisit}.
     */
    abstract boolean shouldVisit(String qualifiedName);

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
        if (value instanceof StarlarkExportable exportable && !exportable.isExported()) {
          // Unexported StarlarkExportables are not usable and therefore do not need to have docs
          // generated.
          return;
        }
        if (value instanceof StarlarkRuleFunction starlarkRuleFunction) {
          visitRule(qualifiedName, starlarkRuleFunction);
        } else if (value instanceof MacroFunction macroFunction) {
          visitMacroFunction(qualifiedName, macroFunction);
        } else if (value instanceof StarlarkProvider starlarkProvider) {
          visitProvider(qualifiedName, starlarkProvider);
        } else if (value instanceof StarlarkFunction starlarkFunction) {
          visitFunction(qualifiedName, starlarkFunction);
        } else if (value instanceof StarlarkDefinedAspect starlarkDefinedAspect) {
          visitAspect(qualifiedName, starlarkDefinedAspect);
        } else if (value instanceof StarlarkRepoRule starlarkRepoRule) {
          visitRepositoryRule(qualifiedName, starlarkRepoRule.getRepoRule());
        } else if (value instanceof ModuleExtension moduleExtension) {
          visitModuleExtension(qualifiedName, moduleExtension);
        } else {
          maybeVisitOtherSymbol(qualifiedName, value);
          if (value instanceof Structure structure) {
            recurseIntoStructure(
                qualifiedName, structure, /* shouldVisitVerifiedForAncestor= */ true);
          }
        }
      } else if (value instanceof Structure structure) {
        recurseIntoStructure(qualifiedName, structure, /* shouldVisitVerifiedForAncestor= */ false);
      }
      // TODO(b/276733504): should we recurse into dicts to search for documentable values? Note
      // that dicts (unlike structs!) can have reference cycles, so we would need to track the set
      // of traversed entities.
    }

    void visitRule(
        @SuppressWarnings("unused") String qualifiedName,
        @SuppressWarnings("unused") StarlarkRuleFunction value)
        throws ExtractionException {}

    void visitMacroFunction(
        @SuppressWarnings("unused") String qualifiedName,
        @SuppressWarnings("unused") MacroFunction value)
        throws ExtractionException {}

    void visitProvider(
        @SuppressWarnings("unused") String qualifiedName,
        @SuppressWarnings("unused") StarlarkProvider value)
        throws ExtractionException {}

    void visitFunction(
        @SuppressWarnings("unused") String qualifiedName,
        @SuppressWarnings("unused") StarlarkFunction value)
        throws ExtractionException {}

    void visitAspect(
        @SuppressWarnings("unused") String qualifiedName,
        @SuppressWarnings("unused") StarlarkDefinedAspect aspect)
        throws ExtractionException {}

    void visitModuleExtension(
        @SuppressWarnings("unused") String qualifiedName,
        @SuppressWarnings("unused") ModuleExtension moduleExtension)
        throws ExtractionException {}

    void visitRepositoryRule(
        @SuppressWarnings("unused") String qualifiedName,
        @SuppressWarnings("unused") RepoRule repoRule)
        throws ExtractionException {}

    void maybeVisitOtherSymbol(
        @SuppressWarnings("unused") String qualifiedName,
        @SuppressWarnings("unused") Object value) {}

    private void recurseIntoStructure(
        String qualifiedName, Structure structure, boolean shouldVisitVerifiedForAncestor)
        throws ExtractionException {
      for (String fieldName : structure.getFieldNames()) {
        if (ExtractorContext.isPublicName(fieldName)) {
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
                getModule(),
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
    private final Module module;
    private final LinkedHashMap<StarlarkProvider.Key, String> qualifiedNames =
        new LinkedHashMap<>();

    private ProviderQualifiedNameCollector(Module module) {
      this.module = module;
    }

    @Override
    Module getModule() {
      return module;
    }

    /**
     * Builds a map from the keys of the Starlark providers which were walked via {@link #traverse}
     * to the qualified names (including any structs) under which those providers may be accessed by
     * a user of this module.
     *
     * <p>If the same provider is accessible under multiple names, the first documentable name wins.
     */
    ImmutableMap<StarlarkProvider.Key, String> buildQualifiedNames() {
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
    boolean shouldVisit(String qualifiedName) {
      return true;
    }

    @Override
    void visitProvider(String qualifiedName, StarlarkProvider value) {
      qualifiedNames.putIfAbsent(value.getKey(), qualifiedName);
    }
  }

  /** A {@link GlobalsVisitor} which extracts documentation for symbols in this module. */
  private static final class DocumentationExtractor extends GlobalsVisitor {
    private final Module module;
    private final ImmutableMap<String, DocComments> docCommentsMap;
    private final ModuleInfo.Builder moduleInfoBuilder;
    private final Predicate<String> isWantedQualifiedName;
    private final ExtractorContext context;
    private final boolean allowUnusedDocComments;

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
        Module module,
        ImmutableMap<String, DocComments> docCommentsMap,
        ModuleInfo.Builder moduleInfoBuilder,
        Predicate<String> isWantedQualifiedName,
        ExtractorContext context,
        boolean allowUnusedDocComments) {
      this.module = module;
      this.docCommentsMap = docCommentsMap;
      this.moduleInfoBuilder = moduleInfoBuilder;
      this.isWantedQualifiedName = isWantedQualifiedName;
      this.context = context;
      this.allowUnusedDocComments = allowUnusedDocComments;
    }

    @Override
    Module getModule() {
      return module;
    }

    @Override
    boolean shouldVisit(String qualifiedName) {
      return isWantedQualifiedName.test(qualifiedName);
    }

    private void checkNoDocComments(String qualifiedName, String what, String expected)
        throws ExtractionException {
      @Nullable DocComments docComments = docCommentsMap.get(qualifiedName);
      if (docComments != null && !allowUnusedDocComments) {
        throw new ExtractionException(
            module,
            String.format(
                "unexpected doc comment for %s on line %s; API documentation for a %s must be"
                    + " provided in %s",
                qualifiedName, docComments.getStartLocation().line(), what, expected));
      }
    }

    @Override
    void visitFunction(String qualifiedName, StarlarkFunction function) throws ExtractionException {
      checkNoDocComments(qualifiedName, "function", "a docstring at the top of the function body");
      moduleInfoBuilder.addFuncInfo(
          StarlarkFunctionInfoExtractor.fromNameAndFunction(
              qualifiedName, function, context.labelRenderer()));
    }

    @Override
    void visitRule(String qualifiedName, StarlarkRuleFunction ruleFunction)
        throws ExtractionException {
      checkNoDocComments(qualifiedName, "rule", "the doc argument to rule()");
      moduleInfoBuilder.addRuleInfo(
          RuleInfoExtractor.buildRuleInfo(context, qualifiedName, ruleFunction.getRuleClass()));
    }

    @Override
    void visitMacroFunction(String qualifiedName, MacroFunction macroFunction)
        throws ExtractionException {
      checkNoDocComments(qualifiedName, "macro", "the doc argument to macro()");
      MacroInfo.Builder macroInfoBuilder = MacroInfo.newBuilder();
      // Record the name under which this symbol is made accessible, which may differ from the
      // symbol's exported name
      macroInfoBuilder.setMacroName(internalToUnicode(qualifiedName));
      // ... but record the origin rule key for cross references.
      macroInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              .setName(internalToUnicode(macroFunction.getName()))
              .setFile(
                  internalToUnicode(
                      context.labelRenderer().render(macroFunction.getExtensionLabel()))));
      macroFunction
          .getDocumentation()
          .map(StringEncoding::internalToUnicode)
          .ifPresent(macroInfoBuilder::setDocString);

      MacroClass macroClass = macroFunction.getMacroClass();
      if (macroClass.isFinalizer()) {
        macroInfoBuilder.setFinalizer(true);
      }
      // For symbolic macros, always extract non-Starlark attributes (to support inherit_attrs).
      ExtractorContext contextForImplicitMacroAttributes =
          context.extractNativelyDefinedAttrs()
              ? context
              : context.toBuilder().extractNativelyDefinedAttrs(true).build();
      AttributeInfoExtractor.addDocumentableAttributes(
          contextForImplicitMacroAttributes,
          IMPLICIT_MACRO_ATTRIBUTES,
          macroClass.getAttributeProvider().getAttributes(),
          macroInfoBuilder::addAttribute);

      moduleInfoBuilder.addMacroInfo(macroInfoBuilder);
    }

    @Override
    void visitProvider(String qualifiedName, StarlarkProvider provider) throws ExtractionException {
      checkNoDocComments(qualifiedName, "provider", "the doc argument to provider()");
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
      providerInfoBuilder.setProviderName(internalToUnicode(qualifiedName));
      // Record the origin provider key for cross references.
      providerInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              .setName(internalToUnicode(provider.getName()))
              .setFile(
                  internalToUnicode(
                      context.labelRenderer().render(provider.getKey().getExtensionLabel()))));
      provider
          .getDocumentation()
          .map(StringEncoding::internalToUnicode)
          .ifPresent(providerInfoBuilder::setDocString);
      ImmutableMap<String, Optional<String>> schema = provider.getSchema();
      if (schema != null) {
        for (Map.Entry<String, Optional<String>> entry : schema.entrySet()) {
          if (ExtractorContext.isPublicName(entry.getKey())) {
            ProviderFieldInfo.Builder fieldInfoBuilder = ProviderFieldInfo.newBuilder();
            fieldInfoBuilder.setName(internalToUnicode(entry.getKey()));
            entry
                .getValue()
                .map(StringEncoding::internalToUnicode)
                .ifPresent(fieldInfoBuilder::setDocString);
            providerInfoBuilder.addFieldInfo(fieldInfoBuilder.build());
          }
        }
      }
      // TODO(b/276733504): if init is a dict-returning native method (e.g. `dict`), do we document
      // it? (This is very unlikely to be useful at present, and would require parsing annotations
      // on the native method.)
      if (provider.getInit() instanceof StarlarkFunction) {
        providerInfoBuilder.setInit(
            StarlarkFunctionInfoExtractor.fromNameAndFunction(
                qualifiedName, (StarlarkFunction) provider.getInit(), context.labelRenderer()));
      }

      moduleInfoBuilder.addProviderInfo(providerInfoBuilder);
    }

    @Override
    void visitAspect(String qualifiedName, StarlarkDefinedAspect aspect)
        throws ExtractionException {
      checkNoDocComments(qualifiedName, "aspect", "the doc argument to aspect()");
      AspectInfo.Builder aspectInfoBuilder = AspectInfo.newBuilder();
      // Record the name under which this symbol is made accessible, which may differ from the
      // symbol's exported name
      aspectInfoBuilder.setAspectName(internalToUnicode(qualifiedName));
      // ... but record the origin aspect key for cross references.
      aspectInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              .setName(internalToUnicode(aspect.getAspectClass().getExportedName()))
              .setFile(
                  internalToUnicode(
                      context
                          .labelRenderer()
                          .render(aspect.getAspectClass().getExtensionLabel()))));
      aspect
          .getDocumentation()
          .map(StringEncoding::internalToUnicode)
          .ifPresent(aspectInfoBuilder::setDocString);
      switch (aspect.getAttributeAspects()) {
        case FixedListSupplier<String> s:
          for (String aspectAttribute : s.getList()) {
            if (ExtractorContext.isPublicName(aspectAttribute)) {
              aspectInfoBuilder.addAspectAttribute(aspectAttribute);
            }
          }
          break;
        case FunctionSupplier<String> s:
          // TODO(b/394400334): Make {@code attr_aspects} function avaiable in {@code AspectInfo}.
          break;
      }

      AttributeInfoExtractor.addDocumentableAttributes(
          context, ImmutableMap.of(), aspect.getAttributes(), aspectInfoBuilder::addAttribute);
      moduleInfoBuilder.addAspectInfo(aspectInfoBuilder);
    }

    @Override
    void visitModuleExtension(String qualifiedName, ModuleExtension moduleExtension)
        throws ExtractionException {
      checkNoDocComments(
          qualifiedName, "module extension", "the doc argument to module_extension()");
      ModuleExtensionInfo.Builder moduleExtensionInfoBuilder = ModuleExtensionInfo.newBuilder();
      moduleExtensionInfoBuilder.setExtensionName(internalToUnicode(qualifiedName));
      moduleExtensionInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              // TODO(arostovtsev): attempt to retrieve the name under which the module was
              // originally defined so we can call setName() too. The easiest solution might be to
              // make ModuleExtension a StarlarkExportable (partially reverting cl/513213080).
              // Alternatively, we'd need to search the defining module's globals, similarly to what
              // we do in FunctionUtil#getFunctionOriginKey.
              .setFile(
                  internalToUnicode(
                      context.labelRenderer().render(moduleExtension.definingBzlFileLabel()))));
      moduleExtension
          .doc()
          .map(StringEncoding::internalToUnicode)
          .ifPresent(moduleExtensionInfoBuilder::setDocString);
      for (Map.Entry<String, TagClass> entry : moduleExtension.tagClasses().entrySet()) {
        ModuleExtensionTagClassInfo.Builder tagClassInfoBuilder =
            ModuleExtensionTagClassInfo.newBuilder();
        tagClassInfoBuilder.setTagName(internalToUnicode(entry.getKey()));
        entry
            .getValue()
            .doc()
            .map(StringEncoding::internalToUnicode)
            .ifPresent(tagClassInfoBuilder::setDocString);
        AttributeInfoExtractor.addDocumentableAttributes(
            context,
            ImmutableMap.of(),
            entry.getValue().attributes(),
            tagClassInfoBuilder::addAttribute);
        moduleExtensionInfoBuilder.addTagClass(tagClassInfoBuilder);
      }
      moduleInfoBuilder.addModuleExtensionInfo(moduleExtensionInfoBuilder);
    }

    @Override
    protected void visitRepositoryRule(String qualifiedName, RepoRule repoRule)
        throws ExtractionException {
      checkNoDocComments(qualifiedName, "repository rule", "the doc argument to repository_rule()");
      RepositoryRuleInfo.Builder repositoryRuleInfoBuilder = RepositoryRuleInfo.newBuilder();
      repositoryRuleInfoBuilder.setRuleName(internalToUnicode(qualifiedName));
      repoRule
          .doc()
          .map(StringEncoding::internalToUnicode)
          .ifPresent(repositoryRuleInfoBuilder::setDocString);
      repositoryRuleInfoBuilder.setOriginKey(
          OriginKey.newBuilder()
              .setName(internalToUnicode(repoRule.id().ruleName()))
              .setFile(
                  internalToUnicode(context.labelRenderer().render(repoRule.id().bzlFileLabel()))));
      AttributeInfoExtractor.addDocumentableAttributes(
          context,
          IMPLICIT_REPOSITORY_RULE_ATTRIBUTES,
          repoRule.attributes(),
          repositoryRuleInfoBuilder::addAttribute);
      for (String env : repoRule.environ()) {
        repositoryRuleInfoBuilder.addEnviron(internalToUnicode(env));
      }
      moduleInfoBuilder.addRepositoryRuleInfo(repositoryRuleInfoBuilder);
    }

    @Override
    void maybeVisitOtherSymbol(String qualifiedName, Object value) {
      @Nullable DocComments docComments = docCommentsMap.get(qualifiedName);
      if (docComments == null) {
        // Don't emit documentation for symbols without doc comments.
        return;
      }
      moduleInfoBuilder.addStarlarkOtherSymbolInfo(
          StarlarkOtherSymbolInfo.newBuilder()
              .setName(internalToUnicode(qualifiedName))
              .setDoc(internalToUnicode(docComments.getText()))
              .setTypeName(Starlark.type(value)));
    }
  }
}
