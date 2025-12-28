package build.stack.devtools.build.constellate;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkdocextract.ExtractionException;
import com.google.devtools.build.lib.vfs.PathFragment;
import build.stack.devtools.build.constellate.rendering.DocstringParseException;

import build.stack.starlark.v1beta1.StarlarkGrpc.StarlarkImplBase;
import build.stack.starlark.v1beta1.StarlarkProtos.Aspect;
import build.stack.starlark.v1beta1.StarlarkProtos.Attribute;
import build.stack.starlark.v1beta1.StarlarkProtos.Macro;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleExtension;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleExtensionTagClass;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleInfoRequest;
import build.stack.starlark.v1beta1.StarlarkProtos.Module;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleCategory;
import build.stack.starlark.v1beta1.StarlarkProtos.PingRequest;
import build.stack.starlark.v1beta1.StarlarkProtos.PingResponse;
import build.stack.starlark.v1beta1.StarlarkProtos.Provider;
import build.stack.starlark.v1beta1.StarlarkProtos.ProviderField;
import build.stack.starlark.v1beta1.StarlarkProtos.RepositoryRule;
import build.stack.starlark.v1beta1.StarlarkProtos.Rule;
import build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation;

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionTagClassInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;

import build.stack.devtools.build.constellate.rendering.ProtoRenderer;
import build.stack.devtools.build.constellate.StarlarkEvaluator;
import build.stack.devtools.build.constellate.StarlarkEvaluator.StarlarkEvaluationException;
import io.grpc.protobuf.StatusProto;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Timer;
import java.util.TimerTask;
import java.util.UUID;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.ParserInput;

/**
 * A basic implementation of a {@link StarlarkImplBase} service. This server
 * requires keepalive pings to prevent it from shutting down.
 */
final class StarlarkServer extends StarlarkImplBase {
    private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
    private static final int oneMinute = 1000 * 60;

    private StarlarkSemantics semantics; // TODO(pcj): make this final?
    private Timer timer;
    private boolean didPing = false;

    public StarlarkServer(StarlarkSemantics semantics) {
        this.semantics = semantics;

        this.timer = new Timer();
        this.timer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (!didPing) {
                    // System.exit(1);
                }
                didPing = false;
            }
        }, oneMinute, oneMinute);
    }

    @Override
    public void ping(PingRequest request, StreamObserver<PingResponse> pingObserver) {
        this.didPing = true;
        logger.atFinest().log("Got Ping.");

        pingObserver.onNext(PingResponse.newBuilder().build());
        pingObserver.onCompleted();
    }

    @Override
    public void moduleInfo(ModuleInfoRequest request, StreamObserver<Module> moduleObserver) {
        Module.Builder module = Module.newBuilder();

        logger.atInfo().log("Processing module: %s", request.getTargetFileLabel());
        logger.atFine().log("ModuleInfo request details: symbol_names=%s, dep_roots=%s, builtins_bzl_path=%s",
                request.getSymbolNamesList(), request.getDepRootsList(), request.getBuiltinsBzlPath());

        try {
            evalModuleInfo(request, module);
        } catch (InterruptedException e) {
            // Operation cancelled
            moduleObserver.onError(StatusUtils.interruptedError(e.getMessage()));
            return;
        } catch (LabelSyntaxException | IllegalArgumentException e) {
            // Invalid input from client - label format or request parameters
            logger.atInfo().withCause(e).log("Invalid argument in ModuleInfo request: %s",
                    request.getTargetFileLabel());
            moduleObserver.onError(StatusUtils.invalidArgumentError(e));
            return;
        } catch (java.nio.file.NoSuchFileException | java.io.FileNotFoundException e) {
            // File or resource not found
            logger.atInfo().withCause(e).log("File not found in ModuleInfo request: %s", request.getTargetFileLabel());
            moduleObserver.onError(StatusUtils.notFoundError(e));
            return;
        } catch (StarlarkEvaluationException | EvalException e) {
            // User's Starlark code has syntax or runtime errors
            logger.atInfo().withCause(e).log("Starlark evaluation error in ModuleInfo request: %s",
                    request.getTargetFileLabel());
            moduleObserver.onError(StatusUtils.invalidArgumentError(e));
            return;
        } catch (IOException e) {
            // Other I/O errors are internal server errors
            logger.atWarning().withCause(e).log("I/O error in ModuleInfo request: %s", request.getTargetFileLabel());
            moduleObserver.onError(StatusUtils.internalError(e));
            return;
        } catch (Exception e) {
            // Unknown/unexpected errors
            logger.atWarning().withCause(e).log("Unexpected error in ModuleInfo request: %s",
                    request.getTargetFileLabel());
            moduleObserver.onError(StatusUtils.internalError(e));
            return;
        }

        moduleObserver.onNext(module.build());
        moduleObserver.onCompleted();
    }

    private void evalModuleInfo(ModuleInfoRequest request, Module.Builder module) throws InterruptedException,
            IOException, LabelSyntaxException, EvalException, StarlarkEvaluationException {

        String targetFileLabelString;
        String outputPath;
        ImmutableSet<String> symbolNames;
        ImmutableList<String> depRoots;

        if (Strings.isNullOrEmpty(request.getTargetFileLabel())) {
            throw new IllegalArgumentException("Expected a target file label and an output file path.");
        }

        targetFileLabelString = request.getTargetFileLabel();
        symbolNames = ImmutableSet.copyOf(request.getSymbolNamesList());
        depRoots = ImmutableList.copyOf(request.getDepRootsList());

        // Parse the target file label - absolute or relative to request.getRel()
        Label targetFileLabel;
        try {
            if (targetFileLabelString.startsWith("@") || targetFileLabelString.startsWith("//")) {
                // Absolute label
                targetFileLabel = Label.parseCanonical(targetFileLabelString);
            } else {
                // Relative label - resolve against the rel path from the request
                String rel = request.getRel();
                if (rel.isEmpty()) {
                    rel = "//";
                } else if (!rel.startsWith("//")) {
                    rel = "//" + rel;
                }
                targetFileLabel = Label.parseCanonical(rel + ":" + targetFileLabelString);
            }
        } catch (LabelSyntaxException e) {
            throw new LabelSyntaxException(
                    String.format(
                            "Invalid target file label '%s' in ModuleInfoRequest (rel='%s'): %s",
                            targetFileLabelString,
                            request.getRel(),
                            e.getMessage()));
        }

        ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<String, ProviderInfo> providerInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctions = ImmutableMap.builder();
        ImmutableMap.Builder<String, AspectInfo> aspectInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<String, RepositoryRuleInfo> repositoryRuleInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<String, ModuleExtensionInfo> moduleExtensionInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<String, MacroInfo> macroInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<Label, String> moduleDocMap = ImmutableMap.builder();
        ImmutableMap.Builder<Label, Map<String, Object>> globals = ImmutableMap.builder();

        if (!Strings.isNullOrEmpty(request.getBuiltinsBzlPath())) {
            semantics = semantics.toBuilder()
                    .set(BuildLanguageOptions.EXPERIMENTAL_BUILTINS_BZL_PATH, request.getBuiltinsBzlPath()).build();
        }

        // Choose file accessor based on whether module_content is provided
        StarlarkFileAccessor fileAccessor = new FilesystemFileAccessor();
        if (!Strings.isNullOrEmpty(request.getModuleContent())) {
            // Use HybridFileAccessor to provide in-memory content for target file
            // We need to compute the file path that Constellate will use to resolve the
            // label
            // The path is derived from the label's path fragment
            String workspaceRoot = targetFileLabel.getWorkspaceRootForStarlarkOnly(semantics);
            String targetFilePath;
            if (workspaceRoot.isEmpty()) {
                // Local workspace file: //pkg:file.bzl -> pkg/file.bzl
                targetFilePath = targetFileLabel.toPathFragment().toString();
            } else {
                // External workspace file: @repo//pkg:file.bzl -> external/repo/pkg/file.bzl
                targetFilePath = workspaceRoot + "/" + targetFileLabel.toPathFragment().toString();
            }
            fileAccessor = new HybridFileAccessor(
                    targetFilePath,
                    request.getModuleContent(),
                    fileAccessor);
        }

        try {
            StarlarkEvaluator evaluator = new StarlarkEvaluator(semantics, fileAccessor, depRoots);

            Path labelPath = evaluator.pathOfLabel(targetFileLabel);
            ParserInput input = evaluator.getInputSource(labelPath.toString());
            module.setFilename(input.getFile());

            evaluator.eval(
                    input,
                    targetFileLabel,
                    ruleInfoMap,
                    providerInfoMap,
                    userDefinedFunctions,
                    aspectInfoMap,
                    repositoryRuleInfoMap,
                    moduleExtensionInfoMap,
                    macroInfoMap,
                    moduleDocMap,
                    module,
                    globals);

        } catch (StarlarkEvaluator.StarlarkEvaluationException exception) {
            exception.printStackTrace();
            System.err.println("Starlark documentation generation failed: " + exception.getMessage());
            throw exception;
        }

        Map<String, RuleInfo> filteredRuleInfos = ruleInfoMap.build().entrySet().stream()
                .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
                .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
        Map<String, ProviderInfo> filteredProviderInfos = providerInfoMap.build().entrySet().stream()
                .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
                .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
        Map<String, StarlarkFunction> filteredStarlarkFunctions = userDefinedFunctions.build().entrySet().stream()
                .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
                .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
        Map<String, AspectInfo> filteredAspectInfos = aspectInfoMap.build().entrySet().stream()
                .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
                .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
        Map<String, RepositoryRuleInfo> filteredRepositoryRuleInfos = repositoryRuleInfoMap.build().entrySet().stream()
                .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
                .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
        Map<String, ModuleExtensionInfo> filteredModuleExtensionInfos = moduleExtensionInfoMap.build().entrySet()
                .stream()
                .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
                .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));
        Map<String, MacroInfo> filteredMacroInfos = macroInfoMap.build().entrySet().stream()
                .filter(entry -> validSymbolName(symbolNames, entry.getKey()))
                .collect(ImmutableMap.toImmutableMap(Entry::getKey, Entry::getValue));

        ProtoRenderer renderer = new ProtoRenderer();
        renderer.appendRuleInfos(filteredRuleInfos.values())
                .appendProviderInfos(filteredProviderInfos.values())
                .appendStarlarkFunctionInfos(filteredStarlarkFunctions)
                .appendAspectInfos(filteredAspectInfos.values())
                .appendRepositoryRuleInfos(filteredRepositoryRuleInfos.values())
                .appendModuleExtensionInfos(filteredModuleExtensionInfos.values())
                .appendMacroInfos(filteredMacroInfos.values())
                .setModuleDocstring(moduleDocMap.build().get(targetFileLabel));

        module.setCategory(ModuleCategory.LOAD);
        module.setName(request.getTargetFileLabel());
        module.setModuleDocstring(renderer.getModuleDocstring());

        // Add any extraction errors to the module
        for (String error : renderer.getErrors()) {
            module.addError(error);
        }

        // Populate wrapper messages with location information
        populateWrapperMessages(module, renderer);

        // response.setModule(starlarkModule.build());
    }

    private static boolean validSymbolName(ImmutableSet<String> symbolNames, String symbolName) {
        if (symbolNames.isEmpty()) {
            // Symbols prefixed with an underscore are private, and thus, by default,
            // documentation
            // should not be generated for them.
            return !symbolName.startsWith("_");
        } else if (symbolNames.contains(symbolName)) {
            return true;
        } else if (symbolName.contains(".")) {
            return symbolNames.contains(symbolName.substring(0, symbolName.indexOf('.')));
        }
        return false;
    }

    /**
     * Populates wrapper messages (Rule, Provider, Aspect, RepositoryRule,
     * ModuleExtension, Macro) in the
     * Module builder by combining entity info from ProtoRenderer with
     * SymbolLocation data.
     *
     * <p>
     * Wrapper messages provide location information for IDE features like
     * go-to-definition.
     */
    private void populateWrapperMessages(Module.Builder module, ProtoRenderer renderer) {
        // Build a map of symbol names to locations for quick lookup
        Map<String, SymbolLocation> locationMap = new java.util.HashMap<>();
        for (SymbolLocation loc : module.getSymbolLocationList()) {
            locationMap.put(loc.getName(), loc);
        }

        // Populate Rule wrappers
        for (RuleInfo ruleInfo : renderer.getRuleInfos()) {
            Rule.Builder ruleBuilder = Rule.newBuilder()
                    .setInfo(ruleInfo);

            // Add location if available
            SymbolLocation location = locationMap.get(ruleInfo.getRuleName());
            if (location != null) {
                ruleBuilder.setLocation(location);
            }

            // Add attribute wrappers
            for (AttributeInfo attrInfo : ruleInfo.getAttributeList()) {
                Attribute.Builder attrBuilder = Attribute.newBuilder()
                        .setInfo(attrInfo);
                // Note: Attribute locations are not currently tracked in symbol_location
                ruleBuilder.addAttribute(attrBuilder.build());
            }

            module.addRule(ruleBuilder.build());
        }

        // Populate Provider wrappers
        for (ProviderInfo providerInfo : renderer.getProviderInfos()) {
            Provider.Builder providerBuilder = Provider.newBuilder()
                    .setInfo(providerInfo);

            // Add location if available
            SymbolLocation location = locationMap.get(providerInfo.getProviderName());
            if (location != null) {
                providerBuilder.setLocation(location);
            }

            // Add field wrappers
            for (ProviderFieldInfo fieldInfo : providerInfo.getFieldInfoList()) {
                ProviderField.Builder fieldBuilder = ProviderField.newBuilder()
                        .setInfo(fieldInfo);
                // Note: Field locations are not currently tracked in symbol_location
                providerBuilder.addField(fieldBuilder.build());
            }

            module.addProvider(providerBuilder.build());
        }

        // Populate Aspect wrappers
        for (AspectInfo aspectInfo : renderer.getAspectInfos()) {
            Aspect.Builder aspectBuilder = Aspect.newBuilder()
                    .setInfo(aspectInfo);

            // Add location if available
            SymbolLocation location = locationMap.get(aspectInfo.getAspectName());
            if (location != null) {
                aspectBuilder.setLocation(location);
            }

            // Add attribute wrappers
            for (AttributeInfo attrInfo : aspectInfo.getAttributeList()) {
                Attribute.Builder attrBuilder = Attribute.newBuilder()
                        .setInfo(attrInfo);
                // Note: Attribute locations are not currently tracked in symbol_location
                aspectBuilder.addAttribute(attrBuilder.build());
            }

            module.addAspect(aspectBuilder.build());
        }

        // Populate RepositoryRule wrappers
        for (RepositoryRuleInfo repoRuleInfo : renderer.getRepositoryRuleInfos()) {
            RepositoryRule.Builder repoRuleBuilder = RepositoryRule.newBuilder()
                    .setInfo(repoRuleInfo);

            // Add location if available
            SymbolLocation location = locationMap.get(repoRuleInfo.getRuleName());
            if (location != null) {
                repoRuleBuilder.setLocation(location);
            }

            // Add attribute wrappers
            for (AttributeInfo attrInfo : repoRuleInfo.getAttributeList()) {
                Attribute.Builder attrBuilder = Attribute.newBuilder()
                        .setInfo(attrInfo);
                // Note: Attribute locations are not currently tracked in symbol_location
                repoRuleBuilder.addAttribute(attrBuilder.build());
            }

            module.addRepositoryRule(repoRuleBuilder.build());
        }

        // Populate ModuleExtension wrappers
        for (ModuleExtensionInfo extensionInfo : renderer.getModuleExtensionInfos()) {
            ModuleExtension.Builder extensionBuilder = ModuleExtension.newBuilder()
                    .setInfo(extensionInfo);

            // Add location if available
            SymbolLocation location = locationMap.get(extensionInfo.getExtensionName());
            if (location != null) {
                extensionBuilder.setLocation(location);
            }

            // Add tag class wrappers
            for (ModuleExtensionTagClassInfo tagClassInfo : extensionInfo.getTagClassList()) {
                ModuleExtensionTagClass.Builder tagClassBuilder = ModuleExtensionTagClass.newBuilder()
                        .setInfo(tagClassInfo);

                // Note: Tag class locations are not currently tracked separately
                // They could be tracked as "extensionName.tagName" in the future

                // Add attribute wrappers for tag class
                for (AttributeInfo attrInfo : tagClassInfo.getAttributeList()) {
                    Attribute.Builder attrBuilder = Attribute.newBuilder()
                            .setInfo(attrInfo);
                    tagClassBuilder.addAttribute(attrBuilder.build());
                }

                extensionBuilder.addTagClass(tagClassBuilder.build());
            }

            module.addModuleExtension(extensionBuilder.build());
        }

        // Populate Macro wrappers
        for (MacroInfo macroInfo : renderer.getMacroInfos()) {
            Macro.Builder macroBuilder = Macro.newBuilder()
                    .setInfo(macroInfo);

            // Add location if available
            SymbolLocation location = locationMap.get(macroInfo.getMacroName());
            if (location != null) {
                macroBuilder.setLocation(location);
            }

            // Add attribute wrappers
            for (AttributeInfo attrInfo : macroInfo.getAttributeList()) {
                Attribute.Builder attrBuilder = Attribute.newBuilder()
                        .setInfo(attrInfo);
                macroBuilder.addAttribute(attrBuilder.build());
            }

            module.addMacro(macroBuilder.build());
        }
    }

}
