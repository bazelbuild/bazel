// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.License.LicenseParsingException;
import com.google.devtools.build.lib.packages.Package.Builder.GeneratedLabelConflict;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.RuleClass.ParsedAttributeValue;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.StringDictUnaryEntry;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.GlobCriteria;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Functionality to deserialize loaded packages.
 */
public class PackageDeserializer {

  private static final Logger LOG = Logger.getLogger(PackageDeserializer.class.getName());

  // Workaround for Java serialization not allowing to pass in a context manually.
  // volatile is needed to ensure that the objects are published safely.
  // TODO(bazel-team): Subclass ObjectOutputStream to pass through environment variables.
  public static volatile RuleClassProvider defaultRuleClassProvider;
  public static volatile FileSystem defaultDeserializerFileSystem;

  private class Context {
    private final Package.Builder packageBuilder;
    private final Path buildFilePath;

    public Context(Path buildFilePath, Package.Builder packageBuilder) {
      this.buildFilePath = buildFilePath;
      this.packageBuilder = packageBuilder;
    }

    Location deserializeLocation(Build.Location location) {
      return new ExplicitLocation(buildFilePath, location);
    }

    ParsedAttributeValue deserializeAttribute(Type<?> expectedType,
        Build.Attribute attrPb)
        throws PackageDeserializationException {
      Object value = deserializeAttributeValue(expectedType, attrPb);
      return new ParsedAttributeValue(
          attrPb.hasExplicitlySpecified() && attrPb.getExplicitlySpecified(), value,
          deserializeLocation(attrPb.getParseableLocation()));
    }

    void deserializeInputFile(Build.SourceFile sourceFile)
        throws PackageDeserializationException {
      InputFile inputFile;
      try {
        inputFile = packageBuilder.createInputFile(
            deserializeLabel(sourceFile.getName()).getName(),
            deserializeLocation(sourceFile.getParseableLocation()));
      } catch (GeneratedLabelConflict e) {
        throw new PackageDeserializationException(e);
      }

      if (!sourceFile.getVisibilityLabelList().isEmpty() || sourceFile.hasLicense()) {
        packageBuilder.setVisibilityAndLicense(inputFile,
            PackageFactory.getVisibility(deserializeLabels(sourceFile.getVisibilityLabelList())),
            deserializeLicense(sourceFile.getLicense()));
      }
    }

    void deserializePackageGroup(Build.PackageGroup packageGroupPb)
        throws PackageDeserializationException {
      List<String> specifications = new ArrayList<>();
      for (String containedPackage : packageGroupPb.getContainedPackageList()) {
        specifications.add("//" + containedPackage);
      }

      try {
        packageBuilder.addPackageGroup(
            deserializeLabel(packageGroupPb.getName()).getName(),
            specifications,
            deserializeLabels(packageGroupPb.getIncludedPackageGroupList()),
            NullEventHandler.INSTANCE,  // TODO(bazel-team): Handle errors properly
            deserializeLocation(packageGroupPb.getParseableLocation()));
      } catch (Label.SyntaxException | Package.NameConflictException e) {
        throw new PackageDeserializationException(e);
      }
    }

    void deserializeRule(Build.Rule rulePb)
        throws PackageDeserializationException {
      RuleClass ruleClass = ruleClassProvider.getRuleClassMap().get(rulePb.getRuleClass());
      if (ruleClass == null) {
        throw new PackageDeserializationException(
            String.format("Invalid rule class '%s'", ruleClass));
      }

      Map<String, ParsedAttributeValue> attributeValues = new HashMap<>();
      for (Build.Attribute attrPb : rulePb.getAttributeList()) {
        Type<?> type = ruleClass.getAttributeByName(attrPb.getName()).getType();
        attributeValues.put(attrPb.getName(), deserializeAttribute(type, attrPb));
      }

      Label ruleLabel = deserializeLabel(rulePb.getName());
      Location ruleLocation = deserializeLocation(rulePb.getParseableLocation());
      try {
        Rule rule = ruleClass.createRuleWithParsedAttributeValues(
            ruleLabel, packageBuilder, ruleLocation, attributeValues,
            NullEventHandler.INSTANCE);
        packageBuilder.addRule(rule);

        Preconditions.checkState(!rule.containsErrors());
      } catch (NameConflictException | SyntaxException e) {
        throw new PackageDeserializationException(e);
      }
    }
  }

  private final FileSystem fileSystem;
  private final RuleClassProvider ruleClassProvider;

  @Immutable
  private static final class ExplicitLocation extends Location {
    private final PathFragment path;
    private final int startLine;
    private final int startColumn;
    private final int endLine;
    private final int endColumn;

    private ExplicitLocation(Path path, Build.Location location) {
      super(
          location.hasStartOffset() && location.hasEndOffset() ? location.getStartOffset() : 0,
          location.hasStartOffset() && location.hasEndOffset() ? location.getEndOffset() : 0);
      this.path = path.asFragment();
      if (location.hasStartLine() && location.hasStartColumn()
          && location.hasEndLine() && location.hasEndColumn()) {
        this.startLine = location.getStartLine();
        this.startColumn = location.getStartColumn();
        this.endLine = location.getEndLine();
        this.endColumn = location.getEndColumn();
      } else {
        this.startLine = 0;
        this.startColumn = 0;
        this.endLine = 0;
        this.endColumn = 0;
      }
    }

    @Override
    public PathFragment getPath() {
      return path;
    }

    @Override
    public LineAndColumn getStartLineAndColumn() {
      return new LineAndColumn(startLine, startColumn);
    }

    @Override
    public LineAndColumn getEndLineAndColumn() {
      return new LineAndColumn(endLine, endColumn);
    }
  }

  public PackageDeserializer(FileSystem fileSystem, RuleClassProvider ruleClassProvider) {
    if (fileSystem == null) {
      fileSystem = defaultDeserializerFileSystem;
    }
    this.fileSystem = Preconditions.checkNotNull(fileSystem);
    if (ruleClassProvider == null) {
      ruleClassProvider = defaultRuleClassProvider;
    }
    this.ruleClassProvider = Preconditions.checkNotNull(ruleClassProvider);
  }

  /**
   * Exception thrown when something goes wrong during package deserialization.
   */
  public static class PackageDeserializationException extends Exception {
    private PackageDeserializationException(String message) {
      super(message);
    }

    private PackageDeserializationException(String message, Exception reason) {
      super(message, reason);
    }

    private PackageDeserializationException(Exception reason) {
      super(reason);
    }
  }

  private static Label deserializeLabel(String labelName) throws PackageDeserializationException {
    try {
      return Label.parseAbsolute(labelName);
    } catch (Label.SyntaxException e) {
      throw new PackageDeserializationException("Invalid label: " + e.getMessage(), e);
    }
  }

  private static List<Label> deserializeLabels(List<String> labelNames)
      throws PackageDeserializationException {
    ImmutableList.Builder<Label> result = ImmutableList.builder();
    for (String labelName : labelNames) {
      result.add(deserializeLabel(labelName));
    }

    return result.build();
  }

  private static License deserializeLicense(Build.License licensePb)
      throws PackageDeserializationException {
    List<String> licenseStrings = new ArrayList<>();
    licenseStrings.addAll(licensePb.getLicenseTypeList());
    for (String exception : licensePb.getExceptionList()) {
      licenseStrings.add("exception=" + exception);
    }

    try {
      return License.parseLicense(licenseStrings);
    } catch (LicenseParsingException e) {
      throw new PackageDeserializationException(e);
    }
  }

  private static Set<DistributionType> deserializeDistribs(List<String> distributions)
      throws PackageDeserializationException {
    try {
      return License.parseDistributions(distributions);
    } catch (LicenseParsingException e) {
      throw new PackageDeserializationException(e);
    }
  }

  private static TriState deserializeTriStateValue(String value)
      throws PackageDeserializationException {
    if (value.equals("yes")) {
      return TriState.YES;
    } else if (value.equals("no")) {
      return TriState.NO;
    } else if (value.equals("auto")) {
      return TriState.AUTO;
    } else {
      throw new PackageDeserializationException(
          String.format("Invalid tristate value: '%s'", value));
    }
  }

  private static List<FilesetEntry> deserializeFilesetEntries(
      List<Build.FilesetEntry> filesetPbs)
      throws PackageDeserializationException {
    ImmutableList.Builder<FilesetEntry> result = ImmutableList.builder();
    for (Build.FilesetEntry filesetPb : filesetPbs) {
      Label srcLabel = deserializeLabel(filesetPb.getSource());
      List<Label> files =
          filesetPb.getFilesPresent() ? deserializeLabels(filesetPb.getFileList()) : null;
      List<String> excludes =
          filesetPb.getExcludeList().isEmpty()
              ? null : ImmutableList.copyOf(filesetPb.getExcludeList());
      String destDir = filesetPb.getDestinationDirectory();
      FilesetEntry.SymlinkBehavior symlinkBehavior =
          pbToSymlinkBehavior(filesetPb.getSymlinkBehavior());
      String stripPrefix = filesetPb.hasStripPrefix() ? filesetPb.getStripPrefix() : null;

      result.add(
          new FilesetEntry(srcLabel, files, excludes, destDir, symlinkBehavior, stripPrefix));
    }

    return result.build();
  }

  /**
   * Deserialize a package from its representation as a protocol message. The inverse of
   * {@link PackageSerializer#serializePackage}.
   * @throws IOException
   */
  private void deserializeInternal(Build.Package packagePb, StoredEventHandler eventHandler,
      Package.Builder builder, InputStream in) throws PackageDeserializationException, IOException {
    Path buildFile = fileSystem.getPath(packagePb.getBuildFilePath());
    Preconditions.checkNotNull(buildFile);
    Context context = new Context(buildFile, builder);
    builder.setFilename(buildFile);

    if (packagePb.hasDefaultVisibilitySet() && packagePb.getDefaultVisibilitySet()) {
      builder.setDefaultVisibility(
          PackageFactory.getVisibility(
              deserializeLabels(packagePb.getDefaultVisibilityLabelList())));
    }

    // It's important to do this after setting the default visibility, since that implicitly sets
    // this bit to true
    builder.setDefaultVisibilitySet(packagePb.getDefaultVisibilitySet());
    if (packagePb.hasDefaultTestonly()) {
      builder.setDefaultTestonly(packagePb.getDefaultTestonly());
    }
    if (packagePb.hasDefaultDeprecation()) {
      builder.setDefaultDeprecation(packagePb.getDefaultDeprecation());
    }

    builder.setDefaultCopts(packagePb.getDefaultCoptList());
    if (packagePb.hasDefaultHdrsCheck()) {
      builder.setDefaultHdrsCheck(packagePb.getDefaultHdrsCheck());
    }
    if (packagePb.hasDefaultLicense()) {
      builder.setDefaultLicense(deserializeLicense(packagePb.getDefaultLicense()));
    }
    builder.setDefaultDistribs(deserializeDistribs(packagePb.getDefaultDistribList()));

    for (String subinclude : packagePb.getSubincludeLabelList()) {
      Label label = deserializeLabel(subinclude);
      builder.addSubinclude(label, null);
    }

    ImmutableList.Builder<Label> skylarkFileDependencies = ImmutableList.builder();
    for (String skylarkFile : packagePb.getSkylarkLabelList()) {
      skylarkFileDependencies.add(deserializeLabel(skylarkFile));
    }
    builder.setSkylarkFileDependencies(skylarkFileDependencies.build());

    MakeEnvironment.Builder makeEnvBuilder = new MakeEnvironment.Builder();
    for (Build.MakeVar makeVar : packagePb.getMakeVariableList()) {
      for (Build.MakeVarBinding binding : makeVar.getBindingList()) {
        makeEnvBuilder.update(
            makeVar.getName(), binding.getValue(), binding.getPlatformSetRegexp());
      }
    }
    builder.setMakeEnv(makeEnvBuilder);

    for (Build.Event event : packagePb.getEventList()) {
      deserializeEvent(context, eventHandler, event);
    }

    if (packagePb.hasContainsErrors() && packagePb.getContainsErrors()) {
      builder.setContainsErrors();
    }
    if (packagePb.hasContainsTemporaryErrors() && packagePb.getContainsTemporaryErrors()) {
      builder.setContainsTemporaryErrors();
    }

    deserializeTargets(in, context);
  }

  private static void deserializeTargets(InputStream in, Context context) throws IOException,
      PackageDeserializationException {
    Build.TargetOrTerminator tot;
    while (!(tot = Build.TargetOrTerminator.parseDelimitedFrom(in)).getIsTerminator()) {
      Build.Target target = tot.getTarget();
      switch (target.getType()) {
        case SOURCE_FILE:
          context.deserializeInputFile(target.getSourceFile());
          break;
        case PACKAGE_GROUP:
          context.deserializePackageGroup(target.getPackageGroup());
          break;
        case RULE:
          context.deserializeRule(target.getRule());
          break;
        default:
          throw new IllegalStateException("Unexpected Target type: " + target.getType());
      }
    }
  }

  /**
   * Deserializes a {@link Package} from {@code in}. The inverse of
   * {@link PackageSerializer#serializePackage}.
   *
   * <p>Expects {@code in} to contain a single
   * {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.Package} message followed
   * by a series of
   * {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.TargetOrTerminator}
   * messages encoding the associated targets.
   *
   * @param in stream to read from
   * @return a new {@link Package} as read from {@code in}
   * @throws PackageDeserializationException on failures deserializing the input
   * @throws IOException on failures reading from {@code in}
   */
  public Package deserialize(InputStream in) throws PackageDeserializationException, IOException {
    try {
      return deserializeInternal(in);
    } catch (PackageDeserializationException | RuntimeException e) {
      LOG.log(Level.WARNING, "Failed to deserialize Package object", e);
      throw e;
    }
  }

  private Package deserializeInternal(InputStream in)
      throws PackageDeserializationException, IOException {
    // Read the initial Package message so we have the data to initialize the builder. We will read
    // the Targets in individually later.
    Build.Package packagePb = Build.Package.parseDelimitedFrom(in);
    Package.Builder builder;
    try {
      builder = new Package.Builder(
          new PackageIdentifier(packagePb.getRepository(), new PathFragment(packagePb.getName())));
    } catch (SyntaxException e) {
      throw new PackageDeserializationException(e);
    }
    StoredEventHandler eventHandler = new StoredEventHandler();
    deserializeInternal(packagePb, eventHandler, builder, in);
    builder.addEvents(eventHandler.getEvents());
    return builder.build();
  }

  private static void deserializeEvent(
      Context context, StoredEventHandler eventHandler, Build.Event event) {
    Location location = null;
    if (event.hasLocation()) {
      location = context.deserializeLocation(event.getLocation());
    }

    String message = event.getMessage();
    switch (event.getKind()) {
      case ERROR: eventHandler.handle(Event.error(location, message)); break;
      case WARNING: eventHandler.handle(Event.warn(location, message)); break;
      case INFO: eventHandler.handle(Event.info(location, message)); break;
      case PROGRESS: eventHandler.handle(Event.progress(location, message)); break;
      default: break;  // Ignore
    }
  }

  private static List<?> deserializeGlobs(List<?> matches,
      Build.Attribute attrPb) {
    if (attrPb.getGlobCriteriaCount() == 0) {
      return matches;
    }

    Builder<GlobCriteria> criteriaBuilder = ImmutableList.builder();
    for (Build.GlobCriteria criteriaPb : attrPb.getGlobCriteriaList()) {
      if (criteriaPb.hasGlob() && criteriaPb.getGlob()) {
        criteriaBuilder.add(GlobCriteria.fromGlobCall(
            ImmutableList.copyOf(criteriaPb.getIncludeList()),
            ImmutableList.copyOf(criteriaPb.getExcludeList())));
      } else {
        criteriaBuilder.add(
            GlobCriteria.fromList(ImmutableList.copyOf(criteriaPb.getIncludeList())));
      }
    }

    @SuppressWarnings({"unchecked", "rawtypes"}) GlobList<?> result =
        new GlobList(criteriaBuilder.build(), matches);
    return result;
  }

  // TODO(bazel-team): Verify that these put sane values in the attribute
  private static Object deserializeAttributeValue(Type<?> expectedType,
      Build.Attribute attrPb)
      throws PackageDeserializationException {
    switch (attrPb.getType()) {
      case INTEGER:
        return attrPb.hasIntValue() ? new Integer(attrPb.getIntValue()) : null;

      case STRING:
        if (!attrPb.hasStringValue()) {
          return null;
        } else if (expectedType == Type.NODEP_LABEL) {
          return deserializeLabel(attrPb.getStringValue());
        } else {
          return attrPb.getStringValue();
        }

      case LABEL:
      case OUTPUT:
        return attrPb.hasStringValue() ? deserializeLabel(attrPb.getStringValue()) : null;

      case STRING_LIST:
        if (expectedType == Type.NODEP_LABEL_LIST) {
          return deserializeGlobs(deserializeLabels(attrPb.getStringListValueList()), attrPb);
        } else {
          return deserializeGlobs(ImmutableList.copyOf(attrPb.getStringListValueList()), attrPb);
        }

      case LABEL_LIST:
      case OUTPUT_LIST:
        return deserializeGlobs(deserializeLabels(attrPb.getStringListValueList()), attrPb);

      case DISTRIBUTION_SET:
        return deserializeDistribs(attrPb.getStringListValueList());

      case LICENSE:
        return attrPb.hasLicense() ? deserializeLicense(attrPb.getLicense()) : null;

      case STRING_DICT: {
        ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
        for (Build.StringDictEntry entry : attrPb.getStringDictValueList()) {
          builder.put(entry.getKey(), entry.getValue());
        }
        return builder.build();
      }

      case STRING_DICT_UNARY: {
        ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
        for (StringDictUnaryEntry entry : attrPb.getStringDictUnaryValueList()) {
          builder.put(entry.getKey(), entry.getValue());
        }
        return builder.build();
      }

      case FILESET_ENTRY_LIST:
        return deserializeFilesetEntries(attrPb.getFilesetListValueList());

      case LABEL_LIST_DICT: {
        ImmutableMap.Builder<String, List<Label>> builder = ImmutableMap.builder();
        for (Build.LabelListDictEntry entry : attrPb.getLabelListDictValueList()) {
          builder.put(entry.getKey(), deserializeLabels(entry.getValueList()));
        }
        return builder.build();
      }

      case STRING_LIST_DICT: {
        ImmutableMap.Builder<String, List<String>> builder = ImmutableMap.builder();
        for (Build.StringListDictEntry entry : attrPb.getStringListDictValueList()) {
          builder.put(entry.getKey(), ImmutableList.copyOf(entry.getValueList()));
        }
        return builder.build();
      }

      case BOOLEAN:
        return attrPb.hasBooleanValue() ? attrPb.getBooleanValue() : null;

      case TRISTATE:
        return attrPb.hasStringValue() ? deserializeTriStateValue(attrPb.getStringValue()) : null;

      default:
          throw new PackageDeserializationException("Invalid discriminator: " + attrPb.getType());
    }
  }

  private static FilesetEntry.SymlinkBehavior pbToSymlinkBehavior(
      Build.FilesetEntry.SymlinkBehavior symlinkBehavior) {
    switch (symlinkBehavior) {
      case COPY:
        return FilesetEntry.SymlinkBehavior.COPY;
      case DEREFERENCE:
        return FilesetEntry.SymlinkBehavior.DEREFERENCE;
      default:
        throw new IllegalStateException();
    }
  }
}
