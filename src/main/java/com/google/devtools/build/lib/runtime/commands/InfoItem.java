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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ProtoUtils;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.AllowedRuleClassInfo;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.AttributeDefinition;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.AttributeValue;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.BuildLanguage;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.RuleDefinition;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An item that is returned by <code>blaze info</code>.
 */
public abstract class InfoItem {
  protected final String name;
  protected final String description;
  protected final boolean hidden;

  protected InfoItem(String name,
      String description,
      boolean hidden) {
    this.name = name;
    this.description = description;
    this.hidden = hidden;
  }

  protected InfoItem(String name,
      String description) {
    this(name, description, false);
  }

  /**
   * The name of the info key.
   */
  public String getName() {
    return name;
  }

  /**
   * The help description of the info key.
   */
  public String getDescription() {
    return description;
  }

  /**
   * Whether the key is printed when "blaze info" is invoked without arguments.
   *
   * <p>This is usually true for info keys that take multiple lines, thus, cannot really be
   * included in the output of argumentless "blaze info".
   */
  public boolean isHidden() {
    return hidden;
  }

  /**
   * Returns the value of the info key. The return value is directly printed to stdout.
   */
  public abstract byte[] get(
      Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
      throws AbruptExitException, InterruptedException;

  protected static byte[] print(Object value) {
    if (value instanceof byte[]) {
      return (byte[]) value;
    }
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    PrintWriter writer = new PrintWriter(new OutputStreamWriter(
        outputStream, StandardCharsets.UTF_8));
    writer.print(value + "\n");
    writer.flush();
    return outputStream.toByteArray();
  }

  /**
   * Info item for the workspace directory.
   */
  public static final class WorkspaceInfoItem extends InfoItem {
    public WorkspaceInfoItem() {
      super("workspace",
          "The working directory of the server.");
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(env);
      return print(env.getRuntime().getWorkspace().getWorkspace());
    }
  }

  /**
   * Info item for the install_base directory.
   */
  public static final class InstallBaseInfoItem extends InfoItem {
    public InstallBaseInfoItem() {
      super("install_base",
          "The installation base directory.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(env);
      return print(env.getRuntime().getWorkspace().getInstallBase());
    }
  }

  /**
   * Info item for the output_base directory.
   */
  public static final class OutputBaseInfoItem extends InfoItem {
    public OutputBaseInfoItem(String productName) {
      super("output_base",
          "A directory for shared " + productName
          + " state as well as tool and strategy specific subdirectories.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(env);
      return print(env.getRuntime().getWorkspace().getOutputBase());
    }
  }

  /**
   * Info item for the execution_root directory.
   */
  public static final class ExecutionRootInfoItem extends InfoItem {
    public ExecutionRootInfoItem() {
      super("execution_root",
          "A directory that makes all input and output files visible to the build.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(env);
      return print(env.getDirectories().getExecRoot(
          configurationSupplier.get().getMainRepositoryName()));
    }
  }

  /**
   * Info item for the output_path directory.
   */
  public static final class OutputPathInfoItem extends InfoItem {
    public OutputPathInfoItem() {
      super("output_path",
          "Output directory",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(env);
      return print(
          env.getDirectories().getOutputPath(configurationSupplier.get().getMainRepositoryName()));
    }
  }

  /**
   * Info item for the {blaze,bazel}-bin directory.
   */
  public static final class BlazeBinInfoItem extends InfoItem {
    public BlazeBinInfoItem(String productName) {
      super(productName + "-bin",
          "Configuration dependent directory for binaries.",
          false);
    }

    // This is one of the three (non-hidden) info items that require a configuration, because the
    // corresponding paths contain the short name. Maybe we should recommend using the symlinks
    // or make them hidden by default?
    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(configurationSupplier);
      return print(configurationSupplier.get().getBinDirectory(RepositoryName.MAIN).getRoot());
    }
  }

  /**
   * Info item for the {blaze,bazel}-genfiles directory.
   */
  public static final class BlazeGenfilesInfoItem extends InfoItem {
    public BlazeGenfilesInfoItem(String productName) {
      super(productName + "-genfiles",
          "Configuration dependent directory for generated files.",
          false);
    }

    // This is one of the three (non-hidden) info items that require a configuration, because the
    // corresponding paths contain the short name. Maybe we should recommend using the symlinks
    // or make them hidden by default?
    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(configurationSupplier);
      return print(configurationSupplier.get().getGenfilesDirectory(RepositoryName.MAIN).getRoot());
    }
  }

  /**
   * Info item for the {blaze,bazel}-testlogs directory.
   */
  public static final class BlazeTestlogsInfoItem extends InfoItem {
    public BlazeTestlogsInfoItem(String productName) {
      super(productName + "-testlogs",
          "Configuration dependent directory for logs from a test run.",
          false);
    }

    // This is one of the three (non-hidden) info items that require a configuration, because the
    // corresponding paths contain the short name. Maybe we should recommend using the symlinks
    // or make them hidden by default?
    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(configurationSupplier);
      return print(configurationSupplier.get().getTestLogsDirectory(RepositoryName.MAIN).getRoot());
    }
  }

  /** Info item for server_log path. */
  public static class ServerLogInfoItem extends InfoItem {
    private static final Logger logger = Logger.getLogger(ServerLogInfoItem.class.getName());

    /**
     * Constructs an info item for the server log path.
     *
     * @param productName name of the tool whose server log path will be queried
     */
    public ServerLogInfoItem(String productName) {
      super("server_log", productName + " server log path", false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      try {
        Optional<Path> path = env.getRuntime().getServerLogPath();
        return print(path.map(Path::toString).orElse(""));
      } catch (IOException e) {
        logger.log(Level.WARNING, "Failed to determine server log location", e);
        return print("UNKNOWN LOG LOCATION");
      }
    }
  }

  /**
   * Info item for release
   */
  public static final class ReleaseInfoItem extends InfoItem {
    public ReleaseInfoItem(String productName) {
      super("release",
          productName + " release identifier",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      return print(BlazeVersionInfo.instance().getReleaseName());
    }
  }

  /**
   * Info item for server_pid
   */
  public static final class ServerPidInfoItem extends InfoItem {
    public ServerPidInfoItem(String productName) {
      super("server_pid",
          productName + " process id",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      return print(ProcessUtils.getpid());
    }
  }

  /**
   * Info item for package_path
   */
  public static final class PackagePathInfoItem extends InfoItem {
    private final OptionsParsingResult commandOptions;

    public PackagePathInfoItem(OptionsParsingResult commandOptions) {
      super("package_path",
          "The search path for resolving package labels.",
          false);
      this.commandOptions = commandOptions;
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(commandOptions);
      PackageCacheOptions packageCacheOptions =
          commandOptions.getOptions(PackageCacheOptions.class);
      return print(Joiner.on(":").join(packageCacheOptions.packagePath));
    }
  }

  private static MemoryUsage getMemoryUsage() {
    MemoryMXBean memBean = ManagementFactory.getMemoryMXBean();
    return memBean.getHeapMemoryUsage();
  }

  /**
   * Info item for the used heap size
   */
  public static final class UsedHeapSizeInfoItem extends InfoItem {
    public UsedHeapSizeInfoItem() {
      super("used-heap-size",
          "The amount of used memory in bytes. Note that this is not a "
          + "good indicator of the actual memory use, as it includes any remaining inaccessible "
          + "memory.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      return print(StringUtilities.prettyPrintBytes(getMemoryUsage().getUsed()));
    }
  }

  /**
   * Info item for the used heap size after garbage collection
   */
  public static final class UsedHeapSizeAfterGcInfoItem extends InfoItem {
    public UsedHeapSizeAfterGcInfoItem() {
      super("used-heap-size-after-gc",
          "The amount of used memory in bytes after a call to System.gc().",
          true);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      System.gc();
      return print(StringUtilities.prettyPrintBytes(getMemoryUsage().getUsed()));
    }
  }

  /**
   * Info item for the committed heap size
   */
  public static final class CommitedHeapSizeInfoItem extends InfoItem {
    public CommitedHeapSizeInfoItem() {
      super("committed-heap-size",
          "The amount of memory in bytes that is committed for the Java virtual machine to use",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      return print(StringUtilities.prettyPrintBytes(getMemoryUsage().getCommitted()));
    }
  }

  /**
   * Info item for the max heap size
   */
  public static final class MaxHeapSizeInfoItem extends InfoItem {
    public MaxHeapSizeInfoItem() {
      super("max-heap-size",
          "The maximum amount of memory in bytes that can be used for memory management.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      return print(StringUtilities.prettyPrintBytes(getMemoryUsage().getMax()));
    }
  }

  /**
   * Info item for the gc-count
   */
  public static final class GcCountInfoItem extends InfoItem {
    public GcCountInfoItem() {
      super("gc-count",
          "Number of garbage collection runs.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      // The documentation is not very clear on what it means to have more than
      // one GC MXBean, so we just sum them up.
      long gcCount = 0;
      for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
        gcCount += gcBean.getCollectionCount();
      }
      return print(gcCount + "");
    }
  }

  /** Info item for the name and version of the Java runtime environment. */
  public static final class JavaRuntimeInfoItem extends InfoItem {
    public JavaRuntimeInfoItem() {
      super("java-runtime", "Name and version of the current Java runtime environment.", false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      return print(
          String.format(
              "%s (build %s) by %s",
              System.getProperty("java.runtime.name", "Unknown runtime"),
              System.getProperty("java.runtime.version", "unknown"),
              System.getProperty("java.vendor", "unknown")));
    }
  }

  /** Info item for the name and version of the Java VM. */
  public static final class JavaVirtualMachineInfoItem extends InfoItem {
    public JavaVirtualMachineInfoItem() {
      super("java-vm", "Name and version of the current Java virtual machine.", false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      return print(
          String.format(
              "%s (build %s, %s) by %s",
              System.getProperty("java.vm.name", "Unknown VM"),
              System.getProperty("java.vm.version", "unknown"),
              System.getProperty("java.vm.info", "unknown"),
              System.getProperty("java.vm.vendor", "unknown")));
    }
  }

  /** Info item for the location of the Java runtime. */
  public static final class JavaHomeInfoItem extends InfoItem {
    public JavaHomeInfoItem() {
      super("java-home", "Location of the current Java runtime.", false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      String javaHome = System.getProperty("java.home");
      if (javaHome == null) {
        return print("unknown");
      }
      // Tunnel through a Path object in order to normalize the representation of the path.
      Path javaHomePath = env.getRuntime().getFileSystem().getPath(javaHome);
      return print(javaHomePath.getPathString());
    }
  }

  /** Info item for the current character encoding settings. */
  public static final class CharacterEncodingInfoItem extends InfoItem {
    public CharacterEncodingInfoItem() {
      super(
          "character-encoding",
          "Information about the character encoding used by the running JVM.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      return print(
          String.format(
              "file.encoding = %s, defaultCharset = %s",
              System.getProperty("file.encoding", "unknown"), Charset.defaultCharset().name()));
    }
  }

  /** Info item for the gc-time */
  public static final class GcTimeInfoItem extends InfoItem {
    public GcTimeInfoItem() {
      super("gc-time",
          "The approximate accumulated time spend on garbage collection.",
          false);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      // The documentation is not very clear on what it means to have more than
      // one GC MXBean, so we just sum them up.
      long gcTime = 0;
      for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
        gcTime += gcBean.getCollectionTime();
      }
      return print(gcTime + "ms");
    }
  }

  /** Info item for the effective current client environment. */
  public static final class ClientEnv extends InfoItem {
    public ClientEnv() {
      super(
          "client-env",
          "The specifications that need to be added to the project-specific rc file to freeze the"
              + " current client environment",
          true);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      String result = "";
      for (Map.Entry<String, String> entry : env.getWhitelistedActionEnv().entrySet()) {
        // TODO(bazel-team): as the syntax of our rc-files does not support to express new-lines in
        // values, we produce syntax errors if the value of the entry contains a newline character.
        result += "build --action_env=" + entry.getKey() + "=" + entry.getValue() + "\n";
      }
      for (Map.Entry<String, String> entry : env.getWhitelistedTestEnv().entrySet()) {
        // TODO(bazel-team): as the syntax of our rc-files does not support to express new-lines in
        // values, we produce syntax errors if the value of the entry contains a newline character.
        result += "build --test_env=" + entry.getKey() + "=" + entry.getValue() + "\n";
      }
      return print(result);
    }
  }

  /**
   * Info item for the default package. It is deprecated, it still works, when explicitly requested,
   * but are not shown by default. It prints multi-line messages and thus don't play well with grep.
   * We don't print them unless explicitly requested.
   *
   * @deprecated
   */
  // TODO(lberki): Try to remove this using an incompatible flag.
  @Deprecated
  public static final class DefaultsPackageInfoItem extends InfoItem {
    public DefaultsPackageInfoItem() {
      super("defaults-package", "Obsolete. Retained for backwards compatibility.", true);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
      checkNotNull(env);
      return print("");
    }
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
        if (!isAbstractRule(otherClass)) {
          info.addAllowedRuleClass(otherClass.getName());
        }
      }
    }

    return info.build();
  }

  private static boolean isAbstractRule(RuleClass c) {
    return c.getName().startsWith("$");
  }

  /**
   * Returns a byte array containing a proto-buffer describing the build language.
   */
  private static byte[] getBuildLanguageDefinition(RuleClassProvider provider) {
    BuildLanguage.Builder resultPb = BuildLanguage.newBuilder();
    Collection<RuleClass> ruleClasses = provider.getRuleClassMap().values();
    for (RuleClass ruleClass : ruleClasses) {
      if (isAbstractRule(ruleClass)) {
        continue;
      }

      RuleDefinition.Builder rulePb = RuleDefinition.newBuilder();
      rulePb.setName(ruleClass.getName());
      for (Attribute attr : ruleClass.getAttributes()) {
        Type<?> t = attr.getType();
        AttributeDefinition.Builder attrPb = AttributeDefinition.newBuilder();
        attrPb.setName(attr.getName());
        attrPb.setType(ProtoUtils.getDiscriminatorFromType(t));
        attrPb.setMandatory(attr.isMandatory());
        attrPb.setAllowEmpty(!attr.isNonEmpty());
        attrPb.setAllowSingleFile(attr.isSingleArtifact());
        attrPb.setConfigurable(attr.isConfigurable());
        attrPb.setCfgIsHost(attr.getTransitionFactory().isHost());

        // Encode default value, if simple.
        Object v = attr.getDefaultValueUnchecked();
        if (!(v == null
            || v instanceof Attribute.ComputedDefault
            || v instanceof Attribute.SkylarkComputedDefaultTemplate
            || v instanceof Attribute.LateBoundDefault
            || v == t.getDefaultValue())) {
          attrPb.setDefault(convertAttrValue(t, v));
        }
        attrPb.setExecutable(attr.isExecutable());
        if (BuildType.isLabelType(t)) {
          attrPb.setAllowedRuleClasses(getAllowedRuleClasses(ruleClasses, attr));
          attrPb.setNodep(t.getLabelClass() == Type.LabelClass.NONDEP_REFERENCE);
        }
        rulePb.addAttribute(attrPb);
      }

      resultPb.addRule(rulePb);
    }

    return resultPb.build().toByteArray();
  }

  // convertAttrValue converts attribute value v of type to t an AttributeValue message.
  private static AttributeValue convertAttrValue(Type<?> t, Object v) {
    AttributeValue.Builder b = AttributeValue.newBuilder();
    if (v instanceof Map) {
      Type.DictType<?, ?> dictType = (Type.DictType<?, ?>) t;
      for (Map.Entry<?, ?> entry : ((Map<?, ?>) v).entrySet()) {
        b.addDictBuilder()
            .setKey(entry.getKey().toString())
            .setValue(convertAttrValue(dictType.getValueType(), entry.getValue()))
            .build();
      }
    } else if (v instanceof List) {
      for (Object elem : (List<?>) v) {
        b.addList(convertAttrValue(t.getListElementType(), elem));
      }
    } else if (t == BuildType.LICENSE) {
      // TODO(adonovan): need dual function of parseLicense.
      // Treat as empty list for now.
    } else if (t == BuildType.DISTRIBUTIONS) {
      // TODO(adonovan): need dual function of parseDistributions.
      // Treat as empty list for now.
    } else if (t == Type.STRING) {
      b.setString((String) v);
    } else if (t == Type.INTEGER) {
      b.setInt((Integer) v);
    } else if (t == Type.BOOLEAN) {
      b.setBool((Boolean) v);
    } else if (t == BuildType.TRISTATE) {
      b.setInt(((TriState) v).toInt());
    } else if (BuildType.isLabelType(t)) { // case order matters!
      b.setString(v.toString());
    } else {
      // No native rule attribute of this type (FilesetEntry?) has a default value.
      throw new IllegalStateException("unexpected type of attribute default value: " + t);
    }
    return b.build();
  }

  /**
   * Info item for the build language. It is deprecated, it still works, when
   * explicitly requested, but are not shown by default. It prints multi-line messages and thus
   * don't play well with grep. We don't print them unless explicitly requested.
   * @Deprecated
   */
  @Deprecated
  public static final class BuildLanguageInfoItem extends InfoItem {
    public BuildLanguageInfoItem() {
      super("build-language",
          "A protobuffer with the build language structure",
          true);
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(env);
      return print(getBuildLanguageDefinition(env.getRuntime().getRuleClassProvider()));
    }
  }

  /**
   * Info item for the default package path. It is deprecated, it still works, when
   * explicitly requested, but are not shown by default. It prints multi-line messages and thus
   * don't play well with grep. We don't print them unless explicitly requested.
   * @deprecated
   */
  @Deprecated
  public static final class DefaultPackagePathInfoItem extends InfoItem {
    private final OptionsParsingResult commandOptions;

    public DefaultPackagePathInfoItem(OptionsParsingResult commandOptions) {
      super("default-package-path",
          "The default package path",
          true);
      this.commandOptions = commandOptions;
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
        throws AbruptExitException {
      checkNotNull(commandOptions);
      return print(Joiner.on(":").join(
          commandOptions.getOptions(PackageCacheOptions.class).packagePath));
    }
  }

  /**
   * Info item for the make environment.
   */
  public static class MakeInfoItem extends InfoItem {
    public MakeInfoItem(String name, String description) {
      super(name, description, false);
    }
    @Override
    public String getDescription() {
      return "Make environment variable '" + name + "'";
    }

    @Override
    public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
      return print(description);
    }
  }
}
