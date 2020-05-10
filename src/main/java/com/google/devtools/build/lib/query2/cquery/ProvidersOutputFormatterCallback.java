package com.google.devtools.build.lib.query2.cquery;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.IOException;
import java.io.OutputStream;

public class ProvidersOutputFormatterCallback extends CqueryThreadsafeCallback {

  ProvidersOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options, OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTarget> accessor) {
    super(eventHandler, options, out, skyframeExecutor, accessor);
  }

  @Override
  public String getName() {
    return "providers";
  }

  @Override
  public void processOutput(Iterable<ConfiguredTarget> partialResult)
      throws IOException, InterruptedException {
    for (ConfiguredTarget configuredTarget : partialResult) {
      if (!(configuredTarget instanceof RuleConfiguredTarget)) {
        continue;
      }

      RuleConfiguredTarget ruleConfiguredTarget = (RuleConfiguredTarget) configuredTarget;

      // BuildConfiguration config =
      //     skyframeExecutor.getConfiguration(eventHandler, configuredTarget.getConfigurationKey());
      StringBuilder output = new StringBuilder();
      // Target actualTarget = accessor.getTargetFromConfiguredTarget(configuredTarget);

      output =
          output
              .append(configuredTarget.getLabel())
              .append(" ")
              .append(ruleConfiguredTarget.getStarlarkProviderKeyStrings().build());

      addResult(output.toString());
    }

  }
}
