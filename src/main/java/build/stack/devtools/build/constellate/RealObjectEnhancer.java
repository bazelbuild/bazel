// Copyright 2025 The Bazel Authors. All rights reserved.
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

package build.stack.devtools.build.constellate;

import build.stack.devtools.build.constellate.rendering.AspectInfoWrapper;
import build.stack.devtools.build.constellate.rendering.MacroInfoWrapper;
import build.stack.devtools.build.constellate.rendering.ModuleExtensionInfoWrapper;
import build.stack.devtools.build.constellate.rendering.ProviderInfoWrapper;
import build.stack.devtools.build.constellate.rendering.RepositoryRuleInfoWrapper;
import build.stack.devtools.build.constellate.rendering.RuleInfoWrapper;
import com.google.common.base.Functions;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.FluentLogger;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions.MacroFunction;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions.StarlarkRuleFunction;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleExtension;
import com.google.devtools.build.lib.bazel.repository.starlark.StarlarkRepositoryModule.StarlarkRepoRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.starlarkdocextract.LabelRenderer;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.OriginKey;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import net.starlark.java.eval.Module;

/**
 * Enhances fake API extraction results with metadata from real evaluated Starlark objects.
 *
 * <p>This class implements a best-effort enhancement pass that runs after successful Starlark
 * module evaluation. It matches real objects (StarlarkRuleFunction, StarlarkProvider, etc.) to
 * the wrapper objects created during fake API interception, then extracts additional metadata
 * like OriginKey that is only available from the real objects.
 *
 * <p>If enhancement fails for any reason, the original fake API results remain intact, ensuring
 * fault tolerance.
 */
public class RealObjectEnhancer {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  private final LabelRenderer labelRenderer;

  public RealObjectEnhancer(Label moduleLabel) {
    // Use default label renderer for simple label display
    this.labelRenderer = LabelRenderer.DEFAULT;
  }

  /**
   * Enhances wrapper objects with metadata from real evaluated objects.
   *
   * @param module the evaluated Starlark module containing real objects
   * @param rules list of rule wrappers to enhance
   * @param providers list of provider wrappers to enhance
   * @param aspects list of aspect wrappers to enhance
   * @param macros list of macro wrappers to enhance
   * @param repositoryRules list of repository rule wrappers to enhance
   * @param moduleExtensions list of module extension wrappers to enhance
   */
  public void enhance(
      Module module,
      List<RuleInfoWrapper> rules,
      List<ProviderInfoWrapper> providers,
      List<AspectInfoWrapper> aspects,
      List<MacroInfoWrapper> macros,
      List<RepositoryRuleInfoWrapper> repositoryRules,
      List<ModuleExtensionInfoWrapper> moduleExtensions) {

    // Build lookup maps from identifier objects to wrappers
    Map<Object, RuleInfoWrapper> rulesByIdentifier = rules.stream()
        .collect(Collectors.toMap(RuleInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<Object, ProviderInfoWrapper> providersByIdentifier = providers.stream()
        .collect(Collectors.toMap(ProviderInfoWrapper::getIdentifier, Functions.identity()));

    Map<Object, AspectInfoWrapper> aspectsByIdentifier = aspects.stream()
        .collect(Collectors.toMap(AspectInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<Object, MacroInfoWrapper> macrosByIdentifier = macros.stream()
        .collect(Collectors.toMap(MacroInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<Object, RepositoryRuleInfoWrapper> repoRulesByIdentifier = repositoryRules.stream()
        .collect(Collectors.toMap(
            RepositoryRuleInfoWrapper::getIdentifierFunction, Functions.identity()));

    Map<Object, ModuleExtensionInfoWrapper> moduleExtsByIdentifier = moduleExtensions.stream()
        .collect(Collectors.toMap(
            ModuleExtensionInfoWrapper::getIdentifierObject, Functions.identity()));

    // Iterate module globals and enhance matching wrappers
    for (var entry : module.getGlobals().entrySet()) {
      String name = entry.getKey();
      Object value = entry.getValue();

      try {
        // Enhance rules
        if (value instanceof StarlarkRuleFunction && rulesByIdentifier.containsKey(value)) {
          enhanceRule(name, (StarlarkRuleFunction) value, rulesByIdentifier.get(value));
        }

        // Enhance providers
        else if (value instanceof StarlarkProvider && providersByIdentifier.containsKey(value)) {
          enhanceProvider(name, (StarlarkProvider) value, providersByIdentifier.get(value));
        }

        // Enhance aspects
        else if (value instanceof StarlarkDefinedAspect && aspectsByIdentifier.containsKey(value)) {
          enhanceAspect(name, (StarlarkDefinedAspect) value, aspectsByIdentifier.get(value));
        }

        // Enhance macros
        else if (value instanceof MacroFunction && macrosByIdentifier.containsKey(value)) {
          enhanceMacro(name, (MacroFunction) value, macrosByIdentifier.get(value));
        }

        // Enhance repository rules
        else if (value instanceof StarlarkRepoRule && repoRulesByIdentifier.containsKey(value)) {
          enhanceRepositoryRule(
              name, ((StarlarkRepoRule) value).getRepoRule(), repoRulesByIdentifier.get(value));
        }

        // Enhance module extensions
        else if (value instanceof ModuleExtension && moduleExtsByIdentifier.containsKey(value)) {
          enhanceModuleExtension(
              name, (ModuleExtension) value, moduleExtsByIdentifier.get(value));
        }
      } catch (Exception e) {
        // Log but don't fail - enhancement is best-effort
        logger.atWarning().withCause(e).log(
            "Failed to enhance %s with real object metadata", name);
      }
    }
  }

  private void enhanceRule(String name, StarlarkRuleFunction ruleFunc, RuleInfoWrapper wrapper) {
    // Extract OriginKey
    OriginKey.Builder originKey = OriginKey.newBuilder()
        .setName(ruleFunc.getRuleClass().getName());

    if (ruleFunc.getRuleClass().isStarlark()) {
      Label extensionLabel = ruleFunc.getRuleClass().getStarlarkExtensionLabel();
      if (extensionLabel != null) {
        originKey.setFile(labelRenderer.render(extensionLabel));
      } else {
        // Fallback for unexported rules
        Label defLabel = ruleFunc.getRuleClass().getRuleDefinitionEnvironmentLabel();
        if (defLabel != null) {
          originKey.setFile(labelRenderer.render(defLabel));
        }
      }
    } else {
      originKey.setFile("<native>");
    }

    wrapper.getRuleInfo().setOriginKey(originKey.build());

    // TODO: Extract advertised providers
    // Requires ProviderNameGroupExtractor to be made public in starlarkdocextract
    // ImmutableSet<StarlarkProviderIdentifier> advertisedProviders =
    //     ruleFunc.getRuleClass().getAdvertisedProviders().getStarlarkProviders();
    // See CONSTELLATE_INTEGRATION_ANALYSIS.md Phase 2 for details

    logger.atFine().log("Enhanced rule %s with OriginKey: %s", name, originKey.getName());
  }

  private void enhanceProvider(
      String name, StarlarkProvider provider, ProviderInfoWrapper wrapper) {
    // Extract OriginKey
    OriginKey.Builder originKey = OriginKey.newBuilder()
        .setName(provider.getName());

    Label extensionLabel = provider.getKey().getExtensionLabel();
    if (extensionLabel != null) {
      originKey.setFile(labelRenderer.render(extensionLabel));
    }

    wrapper.getProviderInfo().setOriginKey(originKey.build());

    // TODO: Extract provider init callback
    // Requires StarlarkFunctionInfoExtractor to be made public in starlarkdocextract
    // if (provider.getInit() instanceof StarlarkFunction) { ... }
    // See CONSTELLATE_INTEGRATION_ANALYSIS.md Phase 3 for details

    // TODO: Extract provider schema field documentation
    // Schema information is already captured during fake API extraction in FakeProviderApi
    // Additional enhancement could extract field docs from provider.getSchema()

    logger.atFine().log("Enhanced provider %s with OriginKey: %s", name, originKey.getName());
  }

  private void enhanceAspect(
      String name, StarlarkDefinedAspect aspect, AspectInfoWrapper wrapper) {
    OriginKey.Builder originKey = OriginKey.newBuilder()
        .setName(aspect.getAspectClass().getExportedName());

    Label extensionLabel = aspect.getAspectClass().getExtensionLabel();
    if (extensionLabel != null) {
      originKey.setFile(labelRenderer.render(extensionLabel));
    }

    wrapper.getAspectInfo().setOriginKey(originKey.build());
    logger.atFine().log("Enhanced aspect %s with OriginKey: %s", name, originKey.getName());
  }

  private void enhanceMacro(String name, MacroFunction macroFunc, MacroInfoWrapper wrapper) {
    OriginKey.Builder originKey = OriginKey.newBuilder()
        .setName(macroFunc.getName());

    Label extensionLabel = macroFunc.getExtensionLabel();
    if (extensionLabel != null) {
      originKey.setFile(labelRenderer.render(extensionLabel));
    }

    wrapper.getMacroInfo().setOriginKey(originKey.build());
    logger.atFine().log("Enhanced macro %s with OriginKey: %s", name, originKey.getName());
  }

  private void enhanceRepositoryRule(
      String name, Object repoRule, RepositoryRuleInfoWrapper wrapper) {
    // Repository rules don't have easy access to OriginKey in the current API
    // This is a placeholder for future enhancement
    logger.atFine().log("Repository rule %s enhancement not yet implemented", name);
  }

  private void enhanceModuleExtension(
      String name, ModuleExtension moduleExt, ModuleExtensionInfoWrapper wrapper) {
    OriginKey.Builder originKey = OriginKey.newBuilder()
        .setName(name);  // Module extensions use the variable name

    Label definingLabel = moduleExt.definingBzlFileLabel();
    if (definingLabel != null) {
      originKey.setFile(labelRenderer.render(definingLabel));
    }

    wrapper.getModuleExtensionInfo().setOriginKey(originKey.build());
    logger.atFine().log("Enhanced module_extension %s with OriginKey: %s", name, name);
  }
}
