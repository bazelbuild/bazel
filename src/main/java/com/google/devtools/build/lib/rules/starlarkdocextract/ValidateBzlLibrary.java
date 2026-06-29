package com.google.devtools.build.lib.rules.starlarkdocextract;

import static com.google.devtools.build.lib.rules.starlarkdocextract.StarlarkDocExtract.verifyModuleDeps;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Sequence;

public final class ValidateBzlLibrary extends NativeAspectClass implements ConfiguredAspectFactory {
  private record StarlarkLibraryInfo(StarlarkInfo starlarkInfo) {
    private static final StarlarkLibraryInfoProvider PROVIDER = new StarlarkLibraryInfoProvider();

    public List<Artifact> getSrcs() {
      try {
        return Sequence.cast(starlarkInfo.getValue("srcs"), Artifact.class, "srcs");
      } catch (EvalException e) {
        throw new IllegalStateException(e);
      }
    }

    public NestedSet<Artifact> getTransitiveSrcs() {
      try {
        return starlarkInfo.getValue("transitive_srcs", Depset.class).getSet(Artifact.class);
      } catch (EvalException | Depset.TypeException e) {
        throw new IllegalStateException(e);
      }
    }

    private static final class StarlarkLibraryInfoProvider
        extends StarlarkProviderWrapper<StarlarkLibraryInfo> {
      public StarlarkLibraryInfoProvider() {
        super(
            BzlLoadValue.keyForBuild(
                Label.parseCanonicalUnchecked("@bazel_skylib+//rules/private:bzl_library.bzl")),
            "StarlarkLibraryInfo");
      }

      @Override
      public StarlarkLibraryInfo wrap(Info value) {
        return new StarlarkLibraryInfo((StarlarkInfo) value);
      }
    }
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return AspectDefinition.builder(this)
        //        .requireStarlarkProviders(
        //            StarlarkProviderIdentifier.forKey(StarlarkLibraryInfo.PROVIDER.getKey()))
        .build();
  }

  @Override
  public ConfiguredAspect create(
      Label targetLabel,
      ConfiguredTarget ct,
      RuleContext context,
      AspectParameters parameters,
      RepositoryName toolsRepository)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    var mainRepoMapping = context.getAnalysisEnvironment().getMainRepoMapping();
    var libraryInfo = ct.get(StarlarkLibraryInfo.PROVIDER);
    if (libraryInfo == null) {
      return ConfiguredAspect.NonApplicableAspect.INSTANCE;
    }
    var modules = new ArrayList<Module>(libraryInfo.getSrcs().size());
    boolean valuesMissing = false;
    for (Artifact src : libraryInfo.getSrcs()) {
      var module = StarlarkDocExtract.loadModule(src, context);
      if (module != null) {
        modules.add(module);
      } else {
        valuesMissing = true;
      }
    }
    if (valuesMissing) {
      throw new CachingAnalysisEnvironment.MissingDepException("module missing");
    }
    String errors =
        modules.stream()
            .map(
                module ->
                    verifyModuleDeps(module, libraryInfo.getTransitiveSrcs(), mainRepoMapping)
                        .map(
                            message ->
                                "in %s: %s"
                                    .formatted(
                                        BazelModuleContext.of(module)
                                            .label()
                                            .getDisplayForm(mainRepoMapping),
                                        message)))
            .flatMap(Optional::stream)
            .collect(joining("\n"));
    var validationOutput =
        context.getDerivedArtifact(
            context
                .getPackageDirectory()
                .getRelative(context.getRule().getName() + "_validation_output.txt"),
            context.getBinDirectory());
    if (errors.isEmpty()) {
      context.registerAction(
          FileWriteAction.createEmptyWithInputs(
              context.getActionOwner(),
              NestedSetBuilder.emptySet(Order.STABLE_ORDER),
              validationOutput));
    } else {
      context.registerAction(
          new FailAction(
              context.getActionOwner(),
              ImmutableList.of(validationOutput),
              errors,
              FailureDetails.FailAction.Code.VALIDATION_FAILURE));
    }
    return ConfiguredAspect.builder(context)
        .addOutputGroup(
            OutputGroupInfo.VALIDATION,
            NestedSetBuilder.create(Order.STABLE_ORDER, validationOutput))
        .build();
  }
}
