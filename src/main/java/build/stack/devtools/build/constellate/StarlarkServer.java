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
import com.google.devtools.build.lib.vfs.PathFragment;
import build.stack.devtools.build.constellate.rendering.DocstringParseException;

import build.stack.starlark.v1beta1.StarlarkGrpc.StarlarkImplBase;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleInfoRequest;
import build.stack.starlark.v1beta1.StarlarkProtos.Module;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleCategory;
import build.stack.starlark.v1beta1.StarlarkProtos.PingRequest;
import build.stack.starlark.v1beta1.StarlarkProtos.PingResponse;

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;

// import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
// import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
// import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
// import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
// import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;
// import build.stack.devtools.build.constellate.rendering.proto.StarlarkServerProtos.ModuleInfoRequest;
// import build.stack.devtools.build.constellate.rendering.proto.StarlarkServerProtos.ModuleInfoResponse;
// import build.stack.devtools.build.constellate.rendering.proto.StarlarkServerProtos.StarlarkModule;

import build.stack.devtools.build.constellate.rendering.ProtoRenderer;
import build.stack.devtools.build.constellate.Constellate;
import build.stack.devtools.build.constellate.Constellate.StarlarkEvaluationException;
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

        logger.atInfo().log("ModuleInfo request: %s", request);

        try {
            evalModuleInfo(request, module);
        } catch (InterruptedException e) {
            moduleObserver.onError(StatusUtils.interruptedError(e.getMessage()));
            return;
        } catch (Exception e) {
            logger.atWarning().withCause(e).log("ModuleInfo request failed: %s", request.getTargetFileLabel());
            moduleObserver.onError(StatusUtils.internalError(e));
            return;
        }

        moduleObserver.onNext(module.build());
        moduleObserver.onCompleted();
    }

    private void evalModuleInfo(ModuleInfoRequest request, Module.Builder module) throws InterruptedException,
            IOException, LabelSyntaxException, EvalException, StarlarkEvaluationException, DocstringParseException {

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

        // TODO (pcj): if Label.parseCommandLineLabel handles absolution labels, why use
        // anything else?
        Label targetFileLabel = Label.parseCommandLineLabel(targetFileLabelString,
                PathFragment.create(request.getRel()));
        // if (!LabelValidator.isAbsolute(targetFileLabelString)) {
        // targetFileLabel = Label.parseCommandLineLabel(targetFileLabelString,
        // PathFragment.create(request.getRel()));
        // } else {
        // targetFileLabel = Label.parseAbsolute(targetFileLabelString,
        // ImmutableMap.of());
        // }

        ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<String, ProviderInfo> providerInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctions = ImmutableMap.builder();
        ImmutableMap.Builder<String, AspectInfo> aspectInfoMap = ImmutableMap.builder();
        ImmutableMap.Builder<Label, String> moduleDocMap = ImmutableMap.builder();
        ImmutableMap.Builder<Label, Map<String, Object>> globals = ImmutableMap.builder();

        if (!Strings.isNullOrEmpty(request.getBuiltinsBzlPath())) {
            semantics = semantics.toBuilder()
                    .set(BuildLanguageOptions.EXPERIMENTAL_BUILTINS_BZL_PATH, request.getBuiltinsBzlPath()).build();
        }
        try {
            Constellate constellate = new Constellate(semantics, new FilesystemFileAccessor(),
                    request.getWorkspaceName(), depRoots);

            Path labelPath = constellate.pathOfLabel(targetFileLabel);
            // FIXME(labelPath will die on relative labels!)
            ParserInput input = constellate.getInputSource(labelPath.toString());
            module.setFilename(input.getFile());

            constellate.eval(
                    input,
                    targetFileLabel,
                    ruleInfoMap,
                    providerInfoMap,
                    userDefinedFunctions,
                    aspectInfoMap,
                    moduleDocMap,
                    module,
                    globals);

        } catch (Constellate.StarlarkEvaluationException exception) {
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

        module.setInfo(new ProtoRenderer().appendRuleInfos(filteredRuleInfos.values())
                .appendProviderInfos(filteredProviderInfos.values())
                .appendStarlarkFunctionInfos(filteredStarlarkFunctions).appendAspectInfos(filteredAspectInfos.values())
                .setModuleDocstring(moduleDocMap.build().get(targetFileLabel)).getModuleInfo().build());
        module.setCategory(ModuleCategory.LOAD);
        module.setName(request.getTargetFileLabel());

        // logger.atFine().log("rule info: %s", ruleInfoMap.build());
        // logger.atFine().log("function info: %s", userDefinedFunctions.build());
        // logger.atFine().log("aspect info: %s", aspectInfoMap.build());

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

}
