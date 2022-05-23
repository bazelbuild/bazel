package com.google.devtools.build.lib.query2.cquery;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.IOException;
import java.io.OutputStream;

public class FilesOutputFormatterCallback extends CqueryThreadsafeCallback {

  private final TopLevelArtifactContext topLevelArtifactContext;

  FilesOutputFormatterCallback(ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<KeyedConfiguredTarget> accessor,
      TopLevelArtifactContext topLevelArtifactContext) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
    this.topLevelArtifactContext = topLevelArtifactContext;
  }

  @Override
  public String getName() {
    return "files";
  }

  @Override
  public void processOutput(Iterable<KeyedConfiguredTarget> partialResult)
      throws IOException, InterruptedException {
    for (KeyedConfiguredTarget keyedTarget : partialResult) {
      ConfiguredTarget target = keyedTarget.getConfiguredTarget();
      if (!TopLevelArtifactHelper.shouldConsiderForDisplay(target)) {
        continue;
      }
      TopLevelArtifactHelper.getAllArtifactsToBuild(target, topLevelArtifactContext)
          .getImportantArtifacts()
          .toList().stream()
          .filter(TopLevelArtifactHelper::shouldDisplay)
          .map(Artifact::getExecPathString)
          .forEach(this::addResult);
    }
  }
}
