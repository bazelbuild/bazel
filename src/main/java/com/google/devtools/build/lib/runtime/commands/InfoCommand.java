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
package com.google.devtools.build.lib.runtime.commands;

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ProtoUtils;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.AllowedRuleClassInfo;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.AttributeDefinition;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.BuildLanguage;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.RuleDefinition;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implementation of 'blaze info'.
 */
@Command(name = "info",
         // TODO(bazel-team): this is not really a build command, but needs access to the
         // configuration options to do its job
         builds = true,
         allowResidue = true,
         binaryStdOut = true,
         help = "resource:info.txt",
         shortDescription = "Displays runtime info about the %{product} server.",
         options = { InfoCommand.Options.class },
         completion = "info-key",
         // We have InfoCommand inherit from {@link BuildCommand} because we want all
         // configuration defaults specified in ~/.blazerc for {@code build} to apply to
         // {@code info} too, even though it doesn't actually do a build.
         //
         // (Ideally there would be a way to make {@code info} inherit just the bare
         // minimum of relevant options from {@code build}, i.e. those that affect the
         // values it prints.  But there's no such mechanism.)
         inherits = { BuildCommand.class })
public class InfoCommand implements BlazeCommand {

  public static class Options extends OptionsBase {
    @Option(name = "show_make_env",
            defaultValue = "false",
            category = "misc",
            help = "Include the \"Make\" environment in the output.")
    public boolean showMakeEnvironment;
  }

  /**
   * Unchecked variant of ExitCausingException. Below, we need to throw from the Supplier interface,
   * which does not allow checked exceptions.
   */
  public static class ExitCausingRuntimeException extends RuntimeException {

    private final ExitCode exitCode;

    public ExitCausingRuntimeException(String message, ExitCode exitCode) {
      super(message);
      this.exitCode = exitCode;
    }

    public ExitCausingRuntimeException(ExitCode exitCode) {
      this.exitCode = exitCode;
    }

    public ExitCode getExitCode() {
      return exitCode;
    }
  }

  private static class HardwiredInfoItem implements BlazeModule.InfoItem {
    private final InfoKey key;
    private final BlazeRuntime runtime;
    private final OptionsProvider commandOptions;

    private HardwiredInfoItem(InfoKey key, BlazeRuntime runtime, OptionsProvider commandOptions) {
      this.key = key;
      this.runtime = runtime;
      this.commandOptions = commandOptions;
    }

    @Override
    public String getName() {
      return key.getName();
    }

    @Override
    public String getDescription() {
      return key.getDescription();
    }

    @Override
    public boolean isHidden() {
      return key.isHidden();
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier) {
      return print(getInfoItem(runtime, key, configurationSupplier, commandOptions));
    }
  }

  private static class MakeInfoItem implements BlazeModule.InfoItem {
    private final String name;
    private final String value;

    private MakeInfoItem(String name, String value) {
      this.name = name;
      this.value = value;
    }

    @Override
    public String getName() {
      return name;
    }

    @Override
    public String getDescription() {
      return "Make environment variable '" + name + "'";
    }

    @Override
    public boolean isHidden() {
      return false;
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier) {
      return print(value);
    }
  }

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser) { }

  @Override
  public ExitCode exec(final CommandEnvironment env, final OptionsProvider optionsProvider) {
    final BlazeRuntime runtime = env.getRuntime();
    env.getReporter().switchToAnsiAllowingHandler();
    Options infoOptions = optionsProvider.getOptions(Options.class);
    OutErr outErr = env.getReporter().getOutErr();
    // Creating a BuildConfiguration is expensive and often unnecessary. Delay the creation until
    // it is needed.
    Supplier<BuildConfiguration> configurationSupplier = new Supplier<BuildConfiguration>() {
      private BuildConfiguration configuration;
      @Override
      public BuildConfiguration get() {
        if (configuration != null) {
          return configuration;
        }
        try {
          // In order to be able to answer configuration-specific queries, we need to setup the
          // package path. Since info inherits all the build options, all the necessary information
          // is available here.
          env.setupPackageCache(
              optionsProvider.getOptions(PackageCacheOptions.class),
              runtime.getDefaultsPackageContent(optionsProvider));
          // TODO(bazel-team): What if there are multiple configurations? [multi-config]
          configuration = env
              .getConfigurations(optionsProvider)
              .getTargetConfigurations().get(0);
          return configuration;
        } catch (InvalidConfigurationException e) {
          env.getReporter().handle(Event.error(e.getMessage()));
          throw new ExitCausingRuntimeException(ExitCode.COMMAND_LINE_ERROR);
        } catch (AbruptExitException e) {
          throw new ExitCausingRuntimeException("unknown error: " + e.getMessage(),
              e.getExitCode());
        } catch (InterruptedException e) {
          env.getReporter().handle(Event.error("interrupted"));
          throw new ExitCausingRuntimeException(ExitCode.INTERRUPTED);
        }
      }
    };

    Map<String, BlazeModule.InfoItem> items = getInfoItemMap(runtime, optionsProvider);

    try {
      if (infoOptions.showMakeEnvironment) {
        Map<String, String> makeEnv = configurationSupplier.get().getMakeEnvironment();
        for (Map.Entry<String, String> entry : makeEnv.entrySet()) {
          BlazeModule.InfoItem item = new MakeInfoItem(entry.getKey(), entry.getValue());
          items.put(item.getName(), item);
        }
      }

      List<String> residue = optionsProvider.getResidue();
      if (residue.size() > 1) {
        env.getReporter().handle(Event.error("at most one key may be specified"));
        return ExitCode.COMMAND_LINE_ERROR;
      }

      String key = residue.size() == 1 ? residue.get(0) : null;
      env.getEventBus().post(new NoBuildEvent());
      if (key != null) { // print just the value for the specified key:
        byte[] value;
        if (items.containsKey(key)) {
          value = items.get(key).get(configurationSupplier);
        } else {
          env.getReporter().handle(Event.error("unknown key: '" + key + "'"));
          return ExitCode.COMMAND_LINE_ERROR;
        }
        try {
          outErr.getOutputStream().write(value);
          outErr.getOutputStream().flush();
        } catch (IOException e) {
          env.getReporter().handle(Event.error("Cannot write info block: " + e.getMessage()));
          return ExitCode.ANALYSIS_FAILURE;
        }
      } else { // print them all
        configurationSupplier.get();  // We'll need this later anyway
        for (BlazeModule.InfoItem infoItem : items.values()) {
          if (infoItem.isHidden()) {
            continue;
          }
          outErr.getOutputStream().write(
              (infoItem.getName() + ": ").getBytes(StandardCharsets.UTF_8));
          outErr.getOutputStream().write(infoItem.get(configurationSupplier));
        }
      }
    } catch (AbruptExitException e) {
      return e.getExitCode();
    } catch (ExitCausingRuntimeException e) {
      return e.getExitCode();
    } catch (IOException e) {
      return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
    }
    return ExitCode.SUCCESS;
  }

  /**
   * Compute and return the info for the given key. Only keys that are not hidden are supported
   * here.
   */
  private static Object getInfoItem(BlazeRuntime runtime, InfoKey key,
      Supplier<BuildConfiguration> configurationSupplier, OptionsProvider options) {
    switch (key) {
      // directories
      case WORKSPACE : return runtime.getDirectories().getWorkspace();
      case INSTALL_BASE : return runtime.getDirectories().getInstallBase();
      case OUTPUT_BASE : return runtime.getDirectories().getOutputBase();
      case EXECUTION_ROOT : return runtime.getDirectories().getExecRoot();
      case OUTPUT_PATH : return runtime.getDirectories().getOutputPath();
      // These are the only (non-hidden) info items that require a configuration, because the
      // corresponding paths contain the short name. Maybe we should recommend using the symlinks
      // or make them hidden by default?
      case BLAZE_BIN : return configurationSupplier.get().getBinDirectory().getPath();
      case BLAZE_GENFILES : return configurationSupplier.get().getGenfilesDirectory().getPath();
      case BLAZE_TESTLOGS : return configurationSupplier.get().getTestLogsDirectory().getPath();

      // logs
      case COMMAND_LOG : return BlazeCommandDispatcher.getCommandLogPath(
          runtime.getDirectories().getOutputBase());
      case MESSAGE_LOG :
        // NB: Duplicated in EventLogModule
        return runtime.getDirectories().getOutputBase().getRelative("message.log");

      // misc
      case RELEASE : return BlazeVersionInfo.instance().getReleaseName();
      case SERVER_PID : return OsUtils.getpid();
      case PACKAGE_PATH : return getPackagePath(options);

      // memory statistics
      case GC_COUNT :
      case GC_TIME :
        // The documentation is not very clear on what it means to have more than
        // one GC MXBean, so we just sum them up.
        int gcCount = 0;
        int gcTime = 0;
        for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
          gcCount += gcBean.getCollectionCount();
          gcTime += gcBean.getCollectionTime();
        }
        if (key == InfoKey.GC_COUNT) {
          return gcCount + "";
        } else {
          return gcTime + "ms";
        }

      case MAX_HEAP_SIZE :
        return StringUtilities.prettyPrintBytes(getMemoryUsage().getMax());
      case USED_HEAP_SIZE :
      case COMMITTED_HEAP_SIZE :
        return StringUtilities.prettyPrintBytes(key == InfoKey.USED_HEAP_SIZE ?
            getMemoryUsage().getUsed() : getMemoryUsage().getCommitted());

      case USED_HEAP_SIZE_AFTER_GC :
        // Note that this info value is not printed by default, but only when explicitly requested.
        System.gc();
        return StringUtilities.prettyPrintBytes(getMemoryUsage().getUsed());

      case DEFAULTS_PACKAGE:
        return runtime.getDefaultsPackageContent();

      case BUILD_LANGUAGE:
        return getBuildLanguageDefinition(runtime.getRuleClassProvider());

      case DEFAULT_PACKAGE_PATH:
        return Joiner.on(":").join(Constants.DEFAULT_PACKAGE_PATH);

      default:
        throw new IllegalArgumentException("missing implementation for " + key);
    }
  }

  private static MemoryUsage getMemoryUsage() {
    MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
    return memBean.getHeapMemoryUsage();
  }

  /**
   * Get the package_path variable for the given set of options.
   */
  private static String getPackagePath(OptionsProvider options) {
    PackageCacheOptions packageCacheOptions =
        options.getOptions(PackageCacheOptions.class);
    return Joiner.on(":").join(packageCacheOptions.packagePath);
  }

  private static AllowedRuleClassInfo getAllowedRuleClasses(
      Collection<RuleClass> ruleClasses, Attribute attr) {
    AllowedRuleClassInfo.Builder info = AllowedRuleClassInfo.newBuilder();
    info.setPolicy(AllowedRuleClassInfo.AllowedRuleClasses.ANY);

    if (attr.isStrictLabelCheckingEnabled()
        && attr.getAllowedRuleClassesPredicate() != Predicates.<RuleClass>alwaysTrue()) {
      info.setPolicy(AllowedRuleClassInfo.AllowedRuleClasses.SPECIFIED);
      Predicate<RuleClass> filter = attr.getAllowedRuleClassesPredicate();
      for (RuleClass otherClass : Iterables.filter(ruleClasses, filter)) {
        if (otherClass.isDocumented()) {
          info.addAllowedRuleClass(otherClass.getName());
        }
      }
    }

    return info.build();
  }

  /**
   * Returns a byte array containing a proto-buffer describing the build language.
   */
  private static byte[] getBuildLanguageDefinition(RuleClassProvider provider) {
    BuildLanguage.Builder resultPb = BuildLanguage.newBuilder();
    Collection<RuleClass> ruleClasses = provider.getRuleClassMap().values();
    for (RuleClass ruleClass : ruleClasses) {
      if (!ruleClass.isDocumented()) {
        continue;
      }

      RuleDefinition.Builder rulePb = RuleDefinition.newBuilder();
      rulePb.setName(ruleClass.getName());
      for (Attribute attr : ruleClass.getAttributes()) {
        if (!attr.isDocumented()) {
          continue;
        }

        AttributeDefinition.Builder attrPb = AttributeDefinition.newBuilder();
        attrPb.setName(attr.getName());
        // The protocol compiler, in its infinite wisdom, generates the field as one of the
        // integer type and the getTypeEnum() method is missing. WTF?
        attrPb.setType(ProtoUtils.getDiscriminatorFromType(attr.getType()));
        attrPb.setMandatory(attr.isMandatory());

        if (BuildType.isLabelType(attr.getType())) {
          attrPb.setAllowedRuleClasses(getAllowedRuleClasses(ruleClasses, attr));
        }

        rulePb.addAttribute(attrPb);
      }

      resultPb.addRule(rulePb);
    }

    return resultPb.build().toByteArray();
  }

  private static byte[] print(Object value) {
    if (value instanceof byte[]) {
      return (byte[]) value;
    }
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    PrintWriter writer = new PrintWriter(outputStream);
    writer.print(value + "\n");
    writer.flush();
    return outputStream.toByteArray();
  }

  static Map<String, BlazeModule.InfoItem> getInfoItemMap(
      BlazeRuntime runtime, OptionsProvider commandOptions) {
    Map<String, BlazeModule.InfoItem> result = new TreeMap<>();  // order by key
    for (BlazeModule module : runtime.getBlazeModules()) {
      for (BlazeModule.InfoItem item : module.getInfoItems()) {
        result.put(item.getName(), item);
      }
    }

    for (InfoKey key : InfoKey.values()) {
      BlazeModule.InfoItem item = new HardwiredInfoItem(key, runtime, commandOptions);
      result.put(item.getName(), item);
    }

    return result;
  }
}
