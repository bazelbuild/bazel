// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.base.Predicates.not;
import static com.google.common.base.Verify.verify;
import static java.util.stream.Collectors.toList;

import android.aapt.pb.internal.ResourcesInternal.CompiledFile;
import com.android.aapt.ConfigurationOuterClass.Configuration;
import com.android.aapt.ConfigurationOuterClass.Configuration.KeysHidden;
import com.android.aapt.ConfigurationOuterClass.Configuration.NavHidden;
import com.android.aapt.ConfigurationOuterClass.Configuration.Orientation;
import com.android.aapt.ConfigurationOuterClass.Configuration.ScreenLayoutLong;
import com.android.aapt.ConfigurationOuterClass.Configuration.ScreenLayoutSize;
import com.android.aapt.ConfigurationOuterClass.Configuration.Touchscreen;
import com.android.aapt.ConfigurationOuterClass.Configuration.UiModeNight;
import com.android.aapt.ConfigurationOuterClass.Configuration.UiModeType;
import com.android.aapt.Resources;
import com.android.aapt.Resources.ConfigValue;
import com.android.aapt.Resources.Package;
import com.android.aapt.Resources.ResourceTable;
import com.android.aapt.Resources.Value;
import com.android.aapt.Resources.XmlNode;
import com.android.ide.common.resources.configuration.CountryCodeQualifier;
import com.android.ide.common.resources.configuration.DensityQualifier;
import com.android.ide.common.resources.configuration.FolderConfiguration;
import com.android.ide.common.resources.configuration.KeyboardStateQualifier;
import com.android.ide.common.resources.configuration.LayoutDirectionQualifier;
import com.android.ide.common.resources.configuration.LocaleQualifier;
import com.android.ide.common.resources.configuration.NavigationMethodQualifier;
import com.android.ide.common.resources.configuration.NavigationStateQualifier;
import com.android.ide.common.resources.configuration.NetworkCodeQualifier;
import com.android.ide.common.resources.configuration.NightModeQualifier;
import com.android.ide.common.resources.configuration.ResourceQualifier;
import com.android.ide.common.resources.configuration.ScreenDimensionQualifier;
import com.android.ide.common.resources.configuration.ScreenHeightQualifier;
import com.android.ide.common.resources.configuration.ScreenOrientationQualifier;
import com.android.ide.common.resources.configuration.ScreenRatioQualifier;
import com.android.ide.common.resources.configuration.ScreenRoundQualifier;
import com.android.ide.common.resources.configuration.ScreenSizeQualifier;
import com.android.ide.common.resources.configuration.ScreenWidthQualifier;
import com.android.ide.common.resources.configuration.SmallestScreenWidthQualifier;
import com.android.ide.common.resources.configuration.TextInputMethodQualifier;
import com.android.ide.common.resources.configuration.TouchScreenQualifier;
import com.android.ide.common.resources.configuration.UiModeQualifier;
import com.android.ide.common.resources.configuration.VersionQualifier;
import com.android.resources.Density;
import com.android.resources.Keyboard;
import com.android.resources.KeyboardState;
import com.android.resources.LayoutDirection;
import com.android.resources.Navigation;
import com.android.resources.NavigationState;
import com.android.resources.NightMode;
import com.android.resources.ResourceType;
import com.android.resources.ScreenOrientation;
import com.android.resources.ScreenRatio;
import com.android.resources.ScreenRound;
import com.android.resources.ScreenSize;
import com.android.resources.TouchScreen;
import com.android.resources.UiMode;
import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.common.io.ByteStreams;
import com.google.common.io.LittleEndianDataInputStream;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.Header;
import com.google.devtools.build.android.resources.Visibility;
import com.google.devtools.build.android.xml.ResourcesAttribute.AttributeType;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import javax.annotation.Nullable;
import javax.annotation.concurrent.NotThreadSafe;

/** Deserializes {@link DataKey}, {@link DataValue} entries from compiled resource files. */
public class AndroidCompiledDataDeserializer implements AndroidDataDeserializer {
  private static final Logger logger =
      Logger.getLogger(AndroidCompiledDataDeserializer.class.getName());

  // Magic numbers from
  // https://android.googlesource.com/platform/frameworks/base.git/+/refs/tags/platform-tools-29.0.2/tools/aapt2/formats.md
  //
  // TODO(b/143116130): aapt2 should just use protobuf directly instead of this custom
  // layer on top.
  private static final int AAPT_CONTAINER_MAGIC = 0x54504141; // "AAPT" as little-endian
  private static final int AAPT_CONTAINER_VERSION = 1;
  private static final int AAPT_CONTAINER_ENTRY_RES_TABLE = 0;
  private static final int AAPT_CONTAINER_ENTRY_RES_FILE = 1;

  // aapt2 insists on aligning things to 4-byte boundaries in *.flat files, though there's no actual
  // performance or maintainability benefit.
  private static final int AAPT_FLAT_FILE_ALIGNMENT = 4;

  private static final ImmutableMap<Configuration.LayoutDirection, LayoutDirection>
      LAYOUT_DIRECTION_MAP =
          ImmutableMap.of(
              Configuration.LayoutDirection.LAYOUT_DIRECTION_LTR,
              LayoutDirection.LTR,
              Configuration.LayoutDirection.LAYOUT_DIRECTION_RTL,
              LayoutDirection.RTL);

  private static final ImmutableMap<Configuration.ScreenLayoutSize, ScreenSize> LAYOUT_SIZE_MAP =
      ImmutableMap.of(
          ScreenLayoutSize.SCREEN_LAYOUT_SIZE_SMALL,
          ScreenSize.SMALL,
          ScreenLayoutSize.SCREEN_LAYOUT_SIZE_NORMAL,
          ScreenSize.NORMAL,
          ScreenLayoutSize.SCREEN_LAYOUT_SIZE_LARGE,
          ScreenSize.LARGE,
          ScreenLayoutSize.SCREEN_LAYOUT_SIZE_XLARGE,
          ScreenSize.XLARGE);

  private static final ImmutableMap<Configuration.ScreenLayoutLong, ScreenRatio> SCREEN_LONG_MAP =
      ImmutableMap.of(
          ScreenLayoutLong.SCREEN_LAYOUT_LONG_LONG,
          ScreenRatio.LONG,
          ScreenLayoutLong.SCREEN_LAYOUT_LONG_NOTLONG,
          ScreenRatio.NOTLONG);

  private static final ImmutableMap<Configuration.ScreenRound, ScreenRound> SCREEN_ROUND_MAP =
      ImmutableMap.of(
          Configuration.ScreenRound.SCREEN_ROUND_ROUND, ScreenRound.ROUND,
          Configuration.ScreenRound.SCREEN_ROUND_NOTROUND, ScreenRound.NOTROUND);

  private static final ImmutableMap<Configuration.Orientation, ScreenOrientation>
      SCREEN_ORIENTATION_MAP =
          ImmutableMap.of(
              Orientation.ORIENTATION_LAND, ScreenOrientation.LANDSCAPE,
              Orientation.ORIENTATION_PORT, ScreenOrientation.PORTRAIT,
              Orientation.ORIENTATION_SQUARE, ScreenOrientation.SQUARE);

  private static final ImmutableMap<UiModeType, UiMode> SCREEN_UI_MODE =
      ImmutableMap.<UiModeType, UiMode>builder()
          .put(UiModeType.UI_MODE_TYPE_APPLIANCE, UiMode.APPLIANCE)
          .put(UiModeType.UI_MODE_TYPE_CAR, UiMode.CAR)
          .put(UiModeType.UI_MODE_TYPE_DESK, UiMode.DESK)
          .put(UiModeType.UI_MODE_TYPE_NORMAL, UiMode.NORMAL)
          .put(UiModeType.UI_MODE_TYPE_TELEVISION, UiMode.TELEVISION)
          .put(UiModeType.UI_MODE_TYPE_VRHEADSET, UiMode.NORMAL)
          .put(UiModeType.UI_MODE_TYPE_WATCH, UiMode.WATCH)
          .build();

  private static final ImmutableMap<Configuration.UiModeNight, NightMode> NIGHT_MODE_MAP =
      ImmutableMap.of(
          UiModeNight.UI_MODE_NIGHT_NIGHT, NightMode.NIGHT,
          UiModeNight.UI_MODE_NIGHT_NOTNIGHT, NightMode.NOTNIGHT);

  private static final ImmutableMap<Configuration.KeysHidden, KeyboardState> KEYBOARD_STATE_MAP =
      ImmutableMap.of(
          KeysHidden.KEYS_HIDDEN_KEYSEXPOSED,
          KeyboardState.EXPOSED,
          KeysHidden.KEYS_HIDDEN_KEYSSOFT,
          KeyboardState.SOFT,
          KeysHidden.KEYS_HIDDEN_KEYSHIDDEN,
          KeyboardState.HIDDEN);

  private static final ImmutableMap<Configuration.Touchscreen, TouchScreen> TOUCH_TYPE_MAP =
      ImmutableMap.of(
          Touchscreen.TOUCHSCREEN_FINGER,
          TouchScreen.FINGER,
          Touchscreen.TOUCHSCREEN_NOTOUCH,
          TouchScreen.NOTOUCH,
          Touchscreen.TOUCHSCREEN_STYLUS,
          TouchScreen.STYLUS);

  private static final ImmutableMap<Configuration.Keyboard, Keyboard> KEYBOARD_MAP =
      ImmutableMap.of(
          Configuration.Keyboard.KEYBOARD_NOKEYS,
          Keyboard.NOKEY,
          Configuration.Keyboard.KEYBOARD_QWERTY,
          Keyboard.QWERTY,
          Configuration.Keyboard.KEYBOARD_TWELVEKEY,
          Keyboard.TWELVEKEY);

  private static final ImmutableMap<Configuration.NavHidden, NavigationState> NAV_STATE_MAP =
      ImmutableMap.of(
          NavHidden.NAV_HIDDEN_NAVHIDDEN,
          NavigationState.HIDDEN,
          NavHidden.NAV_HIDDEN_NAVEXPOSED,
          NavigationState.EXPOSED);

  private static final ImmutableMap<Configuration.Navigation, Navigation> NAVIGATION_MAP =
      ImmutableMap.of(
          Configuration.Navigation.NAVIGATION_DPAD,
          Navigation.DPAD,
          Configuration.Navigation.NAVIGATION_NONAV,
          Navigation.NONAV,
          Configuration.Navigation.NAVIGATION_TRACKBALL,
          Navigation.TRACKBALL,
          Configuration.Navigation.NAVIGATION_WHEEL,
          Navigation.WHEEL);

  private static final ImmutableMap<Integer, Density> DENSITY_MAP =
      ImmutableMap.<Integer, Density>builder()
          .put(0xfffe, Density.ANYDPI)
          .put(0xffff, Density.NODPI)
          .put(120, Density.LOW)
          .put(160, Density.MEDIUM)
          .put(213, Density.TV)
          .put(240, Density.HIGH)
          .put(320, Density.XHIGH)
          .put(480, Density.XXHIGH)
          .put(640, Density.XXXHIGH)
          .build();

  private final boolean includeFileContentsForValidation;

  public static AndroidCompiledDataDeserializer create(boolean includeFileContentsForValidation) {
    return new AndroidCompiledDataDeserializer(includeFileContentsForValidation);
  }

  private AndroidCompiledDataDeserializer(boolean includeFileContentsForValidation) {
    this.includeFileContentsForValidation = includeFileContentsForValidation;
  }

  private static void consumeResourceTable(
      DependencyInfo dependencyInfo,
      KeyValueConsumers consumers,
      ResourceTable resourceTable,
      VisibilityRegistry registry)
      throws UnsupportedEncodingException, InvalidProtocolBufferException {
    List<String> sourcePool =
        decodeSourcePool(resourceTable.getSourcePool().getData().toByteArray());
    ReferenceResolver resolver = ReferenceResolver.asRoot();

    for (Package resourceTablePackage : resourceTable.getPackageList()) {
      ReferenceResolver packageResolver =
          resolver.resolveFor(resourceTablePackage.getPackageName());
      String packageName = resourceTablePackage.getPackageName();

      for (Resources.Type resourceFormatType : resourceTablePackage.getTypeList()) {
        ResourceType resourceType = ResourceType.getEnum(resourceFormatType.getName());

        for (Resources.Entry resource : resourceFormatType.getEntryList()) {
          if (!"android".equals(packageName)) {
            // This means this resource is not in the android sdk, add it to the set.
            for (ConfigValue configValue : resource.getConfigValueList()) {
              FullyQualifiedName fqn =
                  createAndRecordFqn(
                      packageResolver,
                      packageName,
                      resourceType,
                      resource,
                      convertToQualifiers(configValue.getConfig()));
              Visibility visibility =
                  registry.getVisibility(
                      ResourceName.create(packageName, resourceType, resource.getName()));

              int sourceIndex = configValue.getValue().getSource().getPathIdx();
              String source = sourcePool.get(sourceIndex);
              DataSource dataSource = DataSource.of(dependencyInfo, Paths.get(source));

              Value resourceValue = configValue.getValue();
              DataResource dataResource =
                  resourceValue.getItem().hasFile()
                      ? DataValueFile.of(
                          visibility, dataSource, /*fingerprint=*/ null, /*rootXmlNode=*/ null)
                      : DataResourceXml.from(
                          resourceValue, visibility, dataSource, resourceType, packageResolver);

              if (!fqn.isOverwritable()) {
                consumers.combiningConsumer.accept(fqn, dataResource);
              } else {
                consumers.overwritingConsumer.accept(fqn, dataResource);
              }
            }
          } else {
            // In the sdk, just add the fqn for styleables
            createAndRecordFqn(
                packageResolver, packageName, resourceType, resource, ImmutableList.of());
          }
        }
      }
    }
  }

  /** Maintains state for all references in each package of a resource table. */
  // TODO(b/112848607): Remove this!  This machinery is all really for pretty-printing styleables,
  // and only ever used for emitting XML with tools:keep attributes.
  // https://github.com/bazelbuild/bazel/blob/2419d4b2780fc68a0e501c1fab558b045eb054d3/src/tools/android/java/com/google/devtools/build/android/aapt2/ResourceLinker.java#L523
  @NotThreadSafe
  public static class ReferenceResolver {

    enum InlineStatus {
      INLINEABLE,
      INLINED,
    }

    private final Optional<String> packageName;
    private final Map<FullyQualifiedName, InlineStatus> qualifiedReferenceInlineStatus;

    private ReferenceResolver(
        Optional<String> packageName,
        Map<FullyQualifiedName, InlineStatus> qualifiedReferenceInlineStatus) {
      this.packageName = packageName;
      this.qualifiedReferenceInlineStatus = qualifiedReferenceInlineStatus;
    }

    static ReferenceResolver asRoot() {
      return new ReferenceResolver(Optional.empty(), new LinkedHashMap<>());
    }

    public ReferenceResolver resolveFor(String packageName) {
      return new ReferenceResolver(
          Optional.of(packageName).filter(not(String::isEmpty)), qualifiedReferenceInlineStatus);
    }

    public FullyQualifiedName parse(String reference) {
      return FullyQualifiedName.fromReference(reference, packageName);
    }

    public FullyQualifiedName register(FullyQualifiedName fullyQualifiedName) {
      // The default is that the name can be inlined.
      qualifiedReferenceInlineStatus.put(fullyQualifiedName, InlineStatus.INLINEABLE);
      return fullyQualifiedName;
    }

    /** Indicates if a reference can be inlined in a styleable. */
    public boolean shouldInline(FullyQualifiedName reference) {
      // Only inline if it's in the current package.
      if (!reference.isInPackage(packageName.orElse(FullyQualifiedName.DEFAULT_PACKAGE))) {
        return false;
      }

      return InlineStatus.INLINEABLE.equals(qualifiedReferenceInlineStatus.get(reference));
    }

    /** Update the reference's inline state. */
    public FullyQualifiedName markInlined(FullyQualifiedName reference) {
      qualifiedReferenceInlineStatus.put(reference, InlineStatus.INLINED);
      return reference;
    }
  }

  private static FullyQualifiedName createAndRecordFqn(
      ReferenceResolver packageResolver,
      String packageName,
      ResourceType resourceType,
      Resources.Entry resource,
      List<String> qualifiers) {
    final FullyQualifiedName fqn =
        FullyQualifiedName.of(
            packageName.isEmpty() ? FullyQualifiedName.DEFAULT_PACKAGE : packageName,
            qualifiers,
            resourceType,
            resource.getName());
    packageResolver.register(fqn);
    return fqn;
  }

  // TODO(b/146498565): remove this and use 'Configuration' directly, which is typesafe and free.
  private static List<String> convertToQualifiers(Configuration protoConfig) {
    FolderConfiguration configuration = new FolderConfiguration();
    if (protoConfig.getMcc() > 0) {
      configuration.setCountryCodeQualifier(new CountryCodeQualifier(protoConfig.getMcc()));
    }
    // special code for 0, as MNC can be zero
    // https://android.googlesource.com/platform/frameworks/native/+/master/include/android/configuration.h#473
    if (protoConfig.getMnc() != 0) {
      configuration.setNetworkCodeQualifier(
          NetworkCodeQualifier.getQualifier(
              String.format(
                  Locale.US,
                  "mnc%1$03d",
                  protoConfig.getMnc() == 0xffff ? 0 : protoConfig.getMnc())));
    }

    if (!protoConfig.getLocale().isEmpty()) {
      // The proto stores it in a BCP-47 format, but the parser requires a b+ and all the - as +.
      // It's a nice a little impedance mismatch.
      new LocaleQualifier()
          .checkAndSet("b+" + protoConfig.getLocale().replace("-", "+"), configuration);
    }

    if (LAYOUT_DIRECTION_MAP.containsKey(protoConfig.getLayoutDirection())) {
      configuration.setLayoutDirectionQualifier(
          new LayoutDirectionQualifier(LAYOUT_DIRECTION_MAP.get(protoConfig.getLayoutDirection())));
    }

    if (protoConfig.getSmallestScreenWidthDp() > 0) {
      configuration.setSmallestScreenWidthQualifier(
          new SmallestScreenWidthQualifier(protoConfig.getSmallestScreenWidthDp()));
    }

    // screen dimension is defined if one number is greater than 0
    if (Math.max(protoConfig.getScreenHeight(), protoConfig.getScreenWidth()) > 0) {
      configuration.setScreenDimensionQualifier(
          new ScreenDimensionQualifier(
              Math.max(
                  protoConfig.getScreenHeight(),
                  protoConfig.getScreenWidth()), // biggest is always first
              Math.min(protoConfig.getScreenHeight(), protoConfig.getScreenWidth())));
    }

    if (protoConfig.getScreenWidthDp() > 0) {
      configuration.setScreenWidthQualifier(
          new ScreenWidthQualifier(protoConfig.getScreenWidthDp()));
    }

    if (protoConfig.getScreenHeightDp() > 0) {
      configuration.setScreenHeightQualifier(
          new ScreenHeightQualifier(protoConfig.getScreenHeightDp()));
    }

    if (LAYOUT_SIZE_MAP.containsKey(protoConfig.getScreenLayoutSize())) {
      configuration.setScreenSizeQualifier(
          new ScreenSizeQualifier(LAYOUT_SIZE_MAP.get(protoConfig.getScreenLayoutSize())));
    }

    if (SCREEN_LONG_MAP.containsKey(protoConfig.getScreenLayoutLong())) {
      configuration.setScreenRatioQualifier(
          new ScreenRatioQualifier(SCREEN_LONG_MAP.get(protoConfig.getScreenLayoutLong())));
    }

    if (SCREEN_ROUND_MAP.containsKey(protoConfig.getScreenRound())) {
      configuration.setScreenRoundQualifier(
          new ScreenRoundQualifier(SCREEN_ROUND_MAP.get(protoConfig.getScreenRound())));
    }

    if (SCREEN_ORIENTATION_MAP.containsKey(protoConfig.getOrientation())) {
      configuration.setScreenOrientationQualifier(
          new ScreenOrientationQualifier(SCREEN_ORIENTATION_MAP.get(protoConfig.getOrientation())));
    }

    if (SCREEN_UI_MODE.containsKey(protoConfig.getUiModeType())) {
      configuration.setUiModeQualifier(
          new UiModeQualifier(SCREEN_UI_MODE.get(protoConfig.getUiModeType())));
    }

    if (NIGHT_MODE_MAP.containsKey(protoConfig.getUiModeNight())) {
      configuration.setNightModeQualifier(
          new NightModeQualifier(NIGHT_MODE_MAP.get(protoConfig.getUiModeNight())));
    }

    if (DENSITY_MAP.containsKey(protoConfig.getDensity())) {
      configuration.setDensityQualifier(
          new DensityQualifier(DENSITY_MAP.get(protoConfig.getDensity())));
    }

    if (TOUCH_TYPE_MAP.containsKey(protoConfig.getTouchscreen())) {
      configuration.setTouchTypeQualifier(
          new TouchScreenQualifier(TOUCH_TYPE_MAP.get(protoConfig.getTouchscreen())));
    }

    if (KEYBOARD_STATE_MAP.containsKey(protoConfig.getKeysHidden())) {
      configuration.setKeyboardStateQualifier(
          new KeyboardStateQualifier(KEYBOARD_STATE_MAP.get(protoConfig.getKeysHidden())));
    }

    if (KEYBOARD_MAP.containsKey(protoConfig.getKeyboard())) {
      configuration.setTextInputMethodQualifier(
          new TextInputMethodQualifier(KEYBOARD_MAP.get(protoConfig.getKeyboard())));
    }

    if (NAV_STATE_MAP.containsKey(protoConfig.getNavHidden())) {
      configuration.setNavigationStateQualifier(
          new NavigationStateQualifier(NAV_STATE_MAP.get(protoConfig.getNavHidden())));
    }

    if (NAVIGATION_MAP.containsKey(protoConfig.getNavigation())) {
      configuration.setNavigationMethodQualifier(
          new NavigationMethodQualifier(NAVIGATION_MAP.get(protoConfig.getNavigation())));
    }

    if (protoConfig.getSdkVersion() > 0) {
      configuration.setVersionQualifier(new VersionQualifier(protoConfig.getSdkVersion()));
    }

    return Arrays.stream(configuration.getQualifiers())
        .map(ResourceQualifier::getFolderSegment)
        .collect(toList());
  }

  /**
   * Forwards compiled resource data files to consumers, extracting any embedded IDs along the way.
   */
  private static void consumeCompiledFile(
      DependencyInfo dependencyInfo,
      KeyValueConsumers consumers,
      CompiledFileWithData compiledFileWithData,
      VisibilityRegistry registry)
      throws InvalidProtocolBufferException {
    CompiledFile compiledFile = compiledFileWithData.compiledFile();
    Path sourcePath = Paths.get(compiledFile.getSourcePath());
    DataSource dataSource = DataSource.of(dependencyInfo, sourcePath);
    ResourceName resourceName = ResourceName.parse(compiledFile.getResourceName());
    FullyQualifiedName fqn =
        FullyQualifiedName.of(
            resourceName.pkg(),
            convertToQualifiers(compiledFile.getConfig()),
            resourceName.type(),
            resourceName.entry());

    consumers.overwritingConsumer.accept(
        fqn,
        DataValueFile.of(
            registry.getVisibility(resourceName),
            dataSource,
            compiledFileWithData.fingerprint(),
            compiledFileWithData.rootXmlNode()));

    for (CompiledFile.Symbol exportedSymbol : compiledFile.getExportedSymbolList()) {
      if (exportedSymbol.getResourceName().startsWith("android:")) {
        // Skip writing resource xml's for resources in the sdk
        continue;
      }

      ResourceName exportedResourceName = ResourceName.parse(exportedSymbol.getResourceName());
      verify(exportedResourceName.type() == ResourceType.ID);
      FullyQualifiedName symbolFqn =
          FullyQualifiedName.of(
              exportedResourceName.pkg(),
              /*qualifiers=*/ ImmutableList.of(),
              exportedResourceName.type(),
              exportedResourceName.entry());

      DataResourceXml dataResourceXml =
          DataResourceXml.from(
              Value.getDefaultInstance(),
              registry.getVisibility(exportedResourceName),
              dataSource,
              ResourceType.ID,
              /*packageResolver=*/ null);
      consumers.combiningConsumer.accept(symbolFqn, dataResourceXml);
    }
  }

  private static void readAttributesFile(
      DependencyInfo dependencyInfo,
      InputStream resourceFileStream,
      FileSystem fileSystem,
      ParsedAndroidData.KeyValueConsumer<DataKey, DataResource> combine,
      ParsedAndroidData.KeyValueConsumer<DataKey, DataResource> overwrite)
      throws IOException {

    Header header = Header.parseDelimitedFrom(resourceFileStream);
    List<FullyQualifiedName> fullyQualifiedNames = new ArrayList<>();
    for (int i = 0; i < header.getEntryCount(); i++) {
      SerializeFormat.DataKey protoKey =
          SerializeFormat.DataKey.parseDelimitedFrom(resourceFileStream);
      fullyQualifiedNames.add(FullyQualifiedName.fromProto(protoKey));
    }

    DataSourceTable sourceTable =
        DataSourceTable.read(dependencyInfo, resourceFileStream, fileSystem, header);

    for (FullyQualifiedName fullyQualifiedName : fullyQualifiedNames) {
      SerializeFormat.DataValue protoValue =
          SerializeFormat.DataValue.parseDelimitedFrom(resourceFileStream);
      DataSource source = sourceTable.sourceFromId(protoValue.getSourceId());
      DataResourceXml dataResourceXml = (DataResourceXml) DataResourceXml.from(protoValue, source);
      AttributeType attributeType = AttributeType.valueOf(protoValue.getXmlValue().getValueType());

      if (attributeType.isCombining()) {
        combine.accept(fullyQualifiedName, dataResourceXml);
      } else {
        overwrite.accept(fullyQualifiedName, dataResourceXml);
      }
    }
  }

  public static Map<DataKey, DataResource> readAttributes(CompiledResources resources) {
    try (ZipInputStream zipStream = new ZipInputStream(Files.newInputStream(resources.getZip()))) {
      Map<DataKey, DataResource> attributes = new LinkedHashMap<>();
      for (ZipEntry entry = zipStream.getNextEntry();
          entry != null;
          entry = zipStream.getNextEntry()) {
        if (entry.getName().endsWith(CompiledResources.ATTRIBUTES_FILE_EXTENSION)) {
          readAttributesFile(
              // Don't care about origin of ".attributes" values, since they don't feed into field
              // initializers.
              DependencyInfo.UNKNOWN,
              zipStream,
              FileSystems.getDefault(),
              (key, value) ->
                  attributes.put(
                      key,
                      attributes.containsKey(key) ? attributes.get(key).combineWith(value) : value),
              (key, value) ->
                  attributes.put(
                      key,
                      attributes.containsKey(key) ? attributes.get(key).overwrite(value) : value));
        }
      }
      return attributes;
    } catch (IOException e) {
      throw new DeserializationException(e);
    }
  }

  @VisibleForTesting
  public static void readTable(
      DependencyInfo dependencyInfo, InputStream in, KeyValueConsumers consumers)
      throws IOException {
    final ResourceTable resourceTable =
        ResourceTable.parseFrom(in, ExtensionRegistry.getEmptyRegistry());
    consumeResourceTable(dependencyInfo, consumers, resourceTable, new VisibilityRegistry());
  }

  public Map<DataKey, DataResource> read(DependencyInfo dependencyInfo, Path inPath) {
    Map<DataKey, DataResource> resources = new LinkedHashMap<>();
    read(
        dependencyInfo,
        inPath,
        KeyValueConsumers.of(
            resources::put,
            resources::put,
            (key, value) -> {
              throw new IllegalStateException(String.format("Unexpected asset in %s", inPath));
            }));
    return resources;
  }

  @Override
  public void read(DependencyInfo dependencyInfo, Path inPath, KeyValueConsumers consumers) {
    Stopwatch timer = Stopwatch.createStarted();

    try (ZipFile zipFile = new ZipFile(inPath.toFile())) {
      ResourceContainer resourceContainer =
          readResourceContainer(zipFile, includeFileContentsForValidation);
      VisibilityRegistry registry = findPublicResources(resourceContainer);

      for (ResourceTable resourceTable : resourceContainer.resourceTables()) {
        consumeResourceTable(dependencyInfo, consumers, resourceTable, registry);
      }
      for (CompiledFileWithData compiledFile : resourceContainer.compiledFiles()) {
        consumeCompiledFile(dependencyInfo, consumers, compiledFile, registry);
      }

      Enumeration<? extends ZipEntry> resourceFiles = zipFile.entries();
      while (resourceFiles.hasMoreElements()) {
        ZipEntry resourceFile = resourceFiles.nextElement();
        String fileZipPath = resourceFile.getName();

        try (InputStream resourceFileStream = zipFile.getInputStream(resourceFile)) {
          if (fileZipPath.endsWith(CompiledResources.ATTRIBUTES_FILE_EXTENSION)) {
            readAttributesFile(
                dependencyInfo,
                resourceFileStream,
                inPath.getFileSystem(),
                consumers.combiningConsumer,
                consumers.overwritingConsumer);
          }
        }
      }
    } catch (IOException e) {
      throw new DeserializationException("Error deserializing " + inPath, e);
    } finally {
      logger.fine(
          String.format(
              "Deserialized in compiled merged in %sms", timer.elapsed(TimeUnit.MILLISECONDS)));
    }
  }

  private static ResourceContainer readResourceContainer(
      ZipFile zipFile, boolean includeFileContentsForValidation) throws IOException {
    List<ResourceTable> resourceTables = new ArrayList<>();
    List<CompiledFileWithData> compiledFiles = new ArrayList<>();

    Enumeration<? extends ZipEntry> resourceFiles = zipFile.entries();
    while (resourceFiles.hasMoreElements()) {
      ZipEntry resourceFile = resourceFiles.nextElement();
      String fileZipPath = resourceFile.getName();
      if (fileZipPath.endsWith(CompiledResources.ATTRIBUTES_FILE_EXTENSION)) {
        continue;
      }

      try (LittleEndianDataInputStream dataInputStream =
          new LittleEndianDataInputStream(zipFile.getInputStream(resourceFile))) {
        int magic = dataInputStream.readInt();
        verify(
            magic == AAPT_CONTAINER_MAGIC,
            "Unexpected magic number in %s: %s",
            resourceFile,
            magic);
        int version = dataInputStream.readInt();
        verify(
            version == AAPT_CONTAINER_VERSION,
            "Unexpected version number in %s: %s",
            resourceFile,
            version);

        int numberOfEntries = dataInputStream.readInt();
        for (int i = 0; i < numberOfEntries; i++) {
          int entryType = dataInputStream.readInt();
          verify(
              entryType == AAPT_CONTAINER_ENTRY_RES_TABLE
                  || entryType == AAPT_CONTAINER_ENTRY_RES_FILE,
              "Unexpected entry type in %s: %s",
              resourceFile,
              entryType);

          if (entryType == AAPT_CONTAINER_ENTRY_RES_TABLE) {
            long size = dataInputStream.readLong();
            verify(size <= Integer.MAX_VALUE);
            byte[] tableBytes = readBytesAndSkipPadding(dataInputStream, (int) size);

            resourceTables.add(
                ResourceTable.parseFrom(tableBytes, ExtensionRegistry.getEmptyRegistry()));
          } else {
            // useless and wrong "size" data; see
            // https://android-review.googlesource.com/c/platform/frameworks/base/+/1161789
            ByteStreams.skipFully(dataInputStream, 8);
            int headerSize = dataInputStream.readInt();
            long payloadSize = dataInputStream.readLong();
            byte[] headerBytes = readBytesAndSkipPadding(dataInputStream, headerSize);

            CompiledFile compiledFile =
                CompiledFile.parseFrom(headerBytes, ExtensionRegistry.getEmptyRegistry());

            HashCode fingerprint;
            XmlNode rootXmlNode;
            if (includeFileContentsForValidation) {
              byte[] payloadBytes = readBytesAndSkipPadding(dataInputStream, (int) payloadSize);
              fingerprint = Hashing.goodFastHash(/*minimumBits=*/ 64).hashBytes(payloadBytes);
              rootXmlNode =
                  compiledFile.getType() == Resources.FileReference.Type.PROTO_XML
                      ? XmlNode.parseFrom(payloadBytes, ExtensionRegistry.getEmptyRegistry())
                      : null;
            } else {
              ByteStreams.skipFully(
                  dataInputStream, roundUpToNearest(payloadSize, AAPT_FLAT_FILE_ALIGNMENT));
              fingerprint = null;
              rootXmlNode = null;
            }

            compiledFiles.add(CompiledFileWithData.create(compiledFile, fingerprint, rootXmlNode));
          }
        }
      }
    }
    return ResourceContainer.create(resourceTables, compiledFiles);
  }

  /** Rounds {@code n} up to the nearest multiple of {@code k}. */
  private static long roundUpToNearest(long n, int k) {
    return ((n + k - 1) / k) * k;
  }

  // TODO(b/26297204): implied "private"ness needs to be segregated by res/ directory; see also
  // b/148110689.
  private static VisibilityRegistry findPublicResources(ResourceContainer resourceContainer) {
    VisibilityRegistry registry = new VisibilityRegistry();
    for (ResourceTable resourceTable : resourceContainer.resourceTables()) {
      for (Package pkg : resourceTable.getPackageList()) {
        for (Resources.Type type : pkg.getTypeList()) {
          ResourceType resourceType = ResourceType.getEnum(type.getName());
          for (Resources.Entry entry : type.getEntryList()) {
            if (entry.getVisibility().getLevel()
                != com.android.aapt.Resources.Visibility.Level.PUBLIC) {
              continue;
            }
            registry.markPublic(
                ResourceName.create(pkg.getPackageName(), resourceType, entry.getName()));
          }
        }
      }
    }
    return registry;
  }

  private static byte[] readBytesAndSkipPadding(LittleEndianDataInputStream input, int size)
      throws IOException {
    byte[] result = new byte[size];
    input.readFully(result);

    long overage = size % AAPT_FLAT_FILE_ALIGNMENT;
    if (overage != 0) {
      ByteStreams.skipFully(input, AAPT_FLAT_FILE_ALIGNMENT - overage);
    }
    return result;
  }

  private static List<String> decodeSourcePool(byte[] bytes) throws UnsupportedEncodingException {
    ByteBuffer byteBuffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

    int stringCount = byteBuffer.getInt(8);
    boolean isUtf8 = (byteBuffer.getInt(16) & (1 << 8)) != 0;
    int stringsStart = byteBuffer.getInt(20);
    // Position the ByteBuffer after the metadata
    byteBuffer.position(28);

    List<String> strings = new ArrayList<>();

    for (int i = 0; i < stringCount; i++) {
      int stringOffset = stringsStart + byteBuffer.getInt();

      if (isUtf8) {
        int characterCount = byteBuffer.get(stringOffset) & 0xFF;
        if ((characterCount & 0x80) != 0) {
          characterCount =
              ((characterCount & 0x7F) << 8) | (byteBuffer.get(stringOffset + 1) & 0xFF);
        }

        stringOffset += (characterCount >= 0x80 ? 2 : 1);

        int length = byteBuffer.get(stringOffset) & 0xFF;
        if ((length & 0x80) != 0) {
          length = ((length & 0x7F) << 8) | (byteBuffer.get(stringOffset + 1) & 0xFF);
        }

        stringOffset += (length >= 0x80 ? 2 : 1);

        strings.add(new String(bytes, stringOffset, length, "UTF8"));
      } else {
        // TODO(b/148817379): this next block of lines is forming an int with holes in it.
        int characterCount = byteBuffer.get(stringOffset) & 0xFFFF;
        if ((characterCount & 0x8000) != 0) {
          characterCount =
              ((characterCount & 0x7FFF) << 16) | (byteBuffer.get(stringOffset + 2) & 0xFFFF);
        }

        stringOffset += 2 * (characterCount >= 0x8000 ? 2 : 1);

        int length = byteBuffer.get(stringOffset) & 0xFFFF;
        if ((length & 0x8000) != 0) {
          length = ((length & 0x7FFF) << 16) | (byteBuffer.get(stringOffset + 2) & 0xFFFF);
        }

        stringOffset += 2 * (length >= 0x8000 ? 2 : 1);

        strings.add(new String(bytes, stringOffset, length, "UTF16"));
      }
    }

    return strings;
  }

  @AutoValue
  abstract static class ResourceContainer {
    abstract ImmutableList<ResourceTable> resourceTables();

    abstract ImmutableList<CompiledFileWithData> compiledFiles();

    static ResourceContainer create(
        List<ResourceTable> resourceTables, List<CompiledFileWithData> compiledFiles) {
      return new AutoValue_AndroidCompiledDataDeserializer_ResourceContainer(
          ImmutableList.copyOf(resourceTables), ImmutableList.copyOf(compiledFiles));
    }
  }

  @AutoValue
  abstract static class CompiledFileWithData {
    abstract CompiledFile compiledFile();

    @Nullable
    abstract HashCode fingerprint();

    @Nullable
    abstract XmlNode rootXmlNode();

    static CompiledFileWithData create(
        CompiledFile compiledFile, @Nullable HashCode fingerprint, @Nullable XmlNode xmlNode) {
      return new AutoValue_AndroidCompiledDataDeserializer_CompiledFileWithData(
          compiledFile, fingerprint, xmlNode);
    }
  }

  private static class VisibilityRegistry {
    private final Set<ResourceName> explicitlyPublicResources = new HashSet<>();

    void markPublic(ResourceName resourceName) {
      explicitlyPublicResources.add(resourceName);
    }

    Visibility getVisibility(ResourceName resourceName) {
      if (!ResourceCompiler.USE_VISIBILITY_FROM_AAPT2) {
        return Visibility.UNKNOWN;
      }
      if (explicitlyPublicResources.isEmpty()) {
        // TODO(b/146647897): make resources private by default
        return Visibility.UNKNOWN;
      }
      return explicitlyPublicResources.contains(resourceName)
          ? Visibility.PUBLIC
          : Visibility.PRIVATE;
    }
  }
}
