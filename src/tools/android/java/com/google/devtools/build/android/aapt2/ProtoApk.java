// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.aapt2;

import static com.google.common.base.Predicates.not;
import static java.util.stream.Collectors.joining;

import com.android.aapt.Resources;
import com.android.aapt.Resources.Array;
import com.android.aapt.Resources.Attribute.Symbol;
import com.android.aapt.Resources.CompoundValue;
import com.android.aapt.Resources.ConfigValue;
import com.android.aapt.Resources.Entry;
import com.android.aapt.Resources.FileReference;
import com.android.aapt.Resources.Item;
import com.android.aapt.Resources.Package;
import com.android.aapt.Resources.Plural;
import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.ResourceTable;
import com.android.aapt.Resources.Style;
import com.android.aapt.Resources.Type;
import com.android.aapt.Resources.Value;
import com.android.aapt.Resources.XmlAttribute;
import com.android.aapt.Resources.XmlElement;
import com.android.aapt.Resources.XmlNamespace;
import com.android.aapt.Resources.XmlNode;
import com.android.resources.ResourceType;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.common.xml.XmlEscapers;
import com.google.devtools.build.android.AndroidResourceOutputs.UniqueZipBuilder;
import com.google.protobuf.ByteString;
import com.google.protobuf.ExtensionRegistry;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;

/**
 * Provides an interface to an apk in proto format. Since the apk is backed by a zip, it is
 * important to close the ProtoApk when done.
 */
public class ProtoApk implements Closeable {

  static final Logger logger = Logger.getLogger(ProtoApk.class.getName());
  private static final String RESOURCE_TABLE = "resources.pb";
  private static final String MANIFEST = "AndroidManifest.xml";

  private static final String RES_DIRECTORY = "res";

  private final URI uri;
  private final FileSystem apkFileSystem;

  private ProtoApk(URI uri, FileSystem apkFileSystem) {
    this.uri = uri;
    this.apkFileSystem = apkFileSystem;
  }

  /** Reads a ProtoApk from a path and verifies that it is in the expected format. */
  public static ProtoApk readFrom(Path apkPath) throws IOException {
    final URI uri = URI.create("jar:" + apkPath.toUri());
    return readFrom(uri);
  }

  private static ProtoApk readFrom(URI uri) throws IOException {
    final FileSystem apkFileSystem = FileSystems.newFileSystem(uri, ImmutableMap.of());
    Preconditions.checkArgument(Files.exists(apkFileSystem.getPath(RESOURCE_TABLE)));
    Preconditions.checkArgument(Files.exists(apkFileSystem.getPath(MANIFEST)));
    return new ProtoApk(URI.create(uri.getSchemeSpecificPart()), apkFileSystem);
  }

  /**
   * Creates a copy of the current apk.
   *
   * @param destination Path to the new apk destination.
   * @param resourceFilter A filter for determining whether a given resource will be included in the
   *     copy.
   * @return The new ProtoApk.
   * @throws IOException when there are issues reading the apk.
   */
  public ProtoApk copy(Path destination, BiPredicate<ResourceType, String> resourceFilter)
      throws IOException {

    final URI dstZipUri = URI.create("jar:" + destination.toUri());
    try (final ZipFile srcZip = new ZipFile(uri.getPath());
        final UniqueZipBuilder dstZip = UniqueZipBuilder.createFor(destination)) {
      final ResourceTable.Builder dstTableBuilder = ResourceTable.newBuilder();
      final ResourceTable resourceTable =
          ResourceTable.parseFrom(
              Files.newInputStream(apkFileSystem.getPath(RESOURCE_TABLE)),
              ExtensionRegistry.getEmptyRegistry());
      dstTableBuilder.setSourcePool(resourceTable.getSourcePool());
      for (Package pkg : resourceTable.getPackageList()) {
        Package dstPkg = copyPackage(resourceFilter, dstZip, pkg);
        dstTableBuilder.addPackage(dstPkg);
      }
      dstZip.addEntry(RESOURCE_TABLE, dstTableBuilder.build().toByteArray(), ZipEntry.DEFLATED);
      srcZip.stream()
          .filter(not(ZipEntry::isDirectory))
          .filter(entry -> !entry.getName().startsWith(RES_DIRECTORY + "/"))
          .filter(entry -> !entry.getName().equals(RESOURCE_TABLE))
          .forEach(
              entry -> {
                try {
                  createDirectories(dstZip, apkFileSystem.getPath(entry.getName()).getParent());
                  try (InputStream srcEntryInputStream = srcZip.getInputStream(entry)) {
                    byte[] content = ByteStreams.toByteArray(srcEntryInputStream);
                    dstZip.addEntry(entry, content);
                  }
                } catch (IOException e) {
                  throw new RuntimeException(e);
                }
              });
    }

    return readFrom(dstZipUri);
  }

  /**
   * Recursively creates all parent directories in {@code zip} before creating {@code directory},
   * similar to {@link Files#createDirectories}.
   */
  private static void createDirectories(UniqueZipBuilder zip, @Nullable Path directory)
      throws IOException {
    if (directory == null) {
      return;
    }
    createDirectories(zip, directory.getParent());
    zip.addDirEntry(directory.toString());
  }

  private Package copyPackage(
      BiPredicate<ResourceType, String> resourceFilter, UniqueZipBuilder dstZip, Package pkg)
      throws IOException {
    Package.Builder dstPkgBuilder = Package.newBuilder(pkg);
    dstPkgBuilder.clearType();
    for (Resources.Type type : pkg.getTypeList()) {
      copyResourceType(resourceFilter, dstZip, dstPkgBuilder, type);
    }
    return dstPkgBuilder.build();
  }

  private void copyResourceType(
      BiPredicate<ResourceType, String> resourceFilter,
      UniqueZipBuilder dstZip,
      Package.Builder dstPkgBuilder,
      Resources.Type type)
      throws IOException {
    Type.Builder dstTypeBuilder = Resources.Type.newBuilder(type);
    dstTypeBuilder.clearEntry();

    ResourceType resourceType = ResourceType.getEnum(type.getName());
    for (Entry entry : type.getEntryList()) {
      if (resourceFilter.test(resourceType, entry.getName())) {
        copyEntry(dstZip, dstTypeBuilder, entry);
      }
    }
    final Resources.Type dstType = dstTypeBuilder.build();
    if (!dstType.getEntryList().isEmpty()) {
      dstPkgBuilder.addType(dstType);
    }
  }

  private void copyEntry(UniqueZipBuilder dstZip, Type.Builder dstTypeBuilder, Entry entry)
      throws IOException {
    dstTypeBuilder.addEntry(Entry.newBuilder(entry));
    for (ConfigValue configValue : entry.getConfigValueList()) {
      if (configValue.hasValue()
          && configValue.getValue().hasItem()
          && configValue.getValue().getItem().hasFile()) {
        final String path = configValue.getValue().getItem().getFile().getPath();
        final Path apkFileSystemPath = apkFileSystem.getPath(path);
        createDirectories(dstZip, apkFileSystemPath.getParent());
        byte[] content = Files.readAllBytes(apkFileSystemPath);
        dstZip.addEntry(path, content, ZipEntry.STORED);
      }
    }
  }

  public XmlNode getManifest() throws IOException {
    try (InputStream in = Files.newInputStream(apkFileSystem.getPath(MANIFEST))) {
      return XmlNode.parseFrom(in, ExtensionRegistry.getEmptyRegistry());
    }
  }

  /** Copy manifest as xml to an external directory. */
  public Path writeManifestAsXmlTo(Path directory) {
    try (XmlWriter out = XmlWriter.openNew(Files.createDirectories(directory).resolve(MANIFEST))) {
      out.write(getManifest());
      return directory.resolve(MANIFEST);
    } catch (IOException e) {
      throw new ProtoApkException(e);
    }
  }

  /** The apk as path. */
  public Path asApkPath() {
    return Paths.get(uri);
  }

  /** Thrown when errors occur during proto apk processing. */
  public static class ProtoApkException extends Aapt2Exception {
    ProtoApkException(IOException e) {
      super(e);
    }
  }

  private static class XmlWriter implements AutoCloseable {
    static final ByteString ANGLE_OPEN = ByteString.copyFrom("<".getBytes(StandardCharsets.UTF_8));
    static final ByteString SPACE = ByteString.copyFrom(" ".getBytes(StandardCharsets.UTF_8));
    static final ByteString ANGLE_CLOSE = ByteString.copyFrom(">".getBytes(StandardCharsets.UTF_8));
    static final ByteString FORWARD_SLASH =
        ByteString.copyFrom("/".getBytes(StandardCharsets.UTF_8));
    static final ByteString XMLNS = ByteString.copyFrom("xmlns:".getBytes(StandardCharsets.UTF_8));
    static final ByteString EQUALS = ByteString.copyFrom("=".getBytes(StandardCharsets.UTF_8));
    static final ByteString QUOTE = ByteString.copyFrom("\"".getBytes(StandardCharsets.UTF_8));
    static final ByteString COLON = ByteString.copyFrom(":".getBytes(StandardCharsets.UTF_8));
    private static final ByteString XML_PRELUDE =
        ByteString.copyFrom(
            "<?xml version=\"1.0\" encoding=\"utf-8\"?>".getBytes(StandardCharsets.UTF_8));

    private final OutputStream out;
    private final Deque<Map<ByteString, ByteString>> namespaceStack;

    static XmlWriter openNew(Path destination) throws IOException {
      return new XmlWriter(Files.newOutputStream(destination, StandardOpenOption.CREATE_NEW));
    }

    private XmlWriter(OutputStream out) {
      this.out = out;
      this.namespaceStack = new ArrayDeque<>();
    }

    public void write(XmlNode node) throws IOException {
      XML_PRELUDE.writeTo(out);
      writeXmlFrom(node);
    }

    private void writeXmlFrom(XmlNode node) throws IOException {
      if (node.hasElement()) {
        writeXmlFrom(node.getElement());
      } else {
        out.write(node.getTextBytes().toByteArray());
      }
    }

    private void writeXmlFrom(XmlElement element) throws IOException {
      ANGLE_OPEN.writeTo(out);
      if (!element.getNamespaceUriBytes().isEmpty()) {
        findNamespacePrefix(element.getNamespaceUriBytes()).writeTo(out);
        COLON.writeTo(out);
      }
      final ByteString name = element.getNameBytes();
      name.writeTo(out);
      final Map<ByteString, ByteString> namespaces = new LinkedHashMap<>();
      for (XmlNamespace namespace : element.getNamespaceDeclarationList()) {
        final ByteString prefix = namespace.getPrefixBytes();
        SPACE.writeTo(out);
        XMLNS.writeTo(out);
        prefix.writeTo(out);
        EQUALS.writeTo(out);
        quote(namespace.getUriBytes());
        namespaces.put(namespace.getUriBytes(), prefix);
      }
      namespaceStack.push(namespaces);
      for (XmlAttribute attribute : element.getAttributeList()) {
        SPACE.writeTo(out);
        if (!attribute.getNamespaceUriBytes().isEmpty()) {
          findNamespacePrefix(attribute.getNamespaceUriBytes()).writeTo(out);
          COLON.writeTo(out);
        }
        attribute.getNameBytes().writeTo(out);
        EQUALS.writeTo(out);
        quote(attribute.getValueBytes());
      }
      if (element.getChildList().isEmpty()) {
        FORWARD_SLASH.writeTo(out);
        ANGLE_CLOSE.writeTo(out);
      } else {
        ANGLE_CLOSE.writeTo(out);
        for (XmlNode child : element.getChildList()) {
          writeXmlFrom(child);
        }
        ANGLE_OPEN.writeTo(out);
        FORWARD_SLASH.writeTo(out);
        name.writeTo(out);
        ANGLE_CLOSE.writeTo(out);
      }
      namespaceStack.pop();
    }

    private void quote(ByteString bytes) throws IOException {
      QUOTE.writeTo(out);
      out.write(
          XmlEscapers.xmlAttributeEscaper()
              .escape(bytes.toStringUtf8())
              .getBytes(StandardCharsets.UTF_8));
      QUOTE.writeTo(out);
    }

    private ByteString findNamespacePrefix(ByteString uri) {
      for (Map<ByteString, ByteString> uriToPrefix : namespaceStack) {
        if (uriToPrefix.containsKey(uri)) {
          return uriToPrefix.get(uri);
        }
      }
      throw new IllegalStateException(
          "Unable to find prefix for "
              + uri.toStringUtf8()
              + " in [ "
              + namespaceStack.stream()
                  .map(Map::keySet)
                  .flatMap(Set::stream)
                  .map(ByteString::toString)
                  .collect(joining(", "))
              + " ]");
    }

    @Override
    public void close() throws IOException {
      out.close();
    }
  }

  /** Traverses the resource table and compiled xml resource using the {@link ResourceVisitor}. */
  public <T extends ResourceVisitor> T visitResources(T visitor) throws IOException {

    // visit manifest
    visitXmlResource(apkFileSystem.getPath(MANIFEST), visitor.enteringManifest());

    // visit resource table and associated files.
    final ResourceTable resourceTable =
        ResourceTable.parseFrom(
            Files.newInputStream(apkFileSystem.getPath(RESOURCE_TABLE)),
            ExtensionRegistry.getEmptyRegistry());

    final List<String> sourcePool =
        resourceTable.hasSourcePool()
            ? decodeSourcePool(resourceTable.getSourcePool().getData().toByteArray())
            : ImmutableList.of();

    for (Package pkg : resourceTable.getPackageList()) {
      ResourcePackageVisitor pkgVisitor =
          visitor.enteringPackage(pkg.getPackageId().getId(), pkg.getPackageName());
      if (pkgVisitor != null) {
        for (Resources.Type type : pkg.getTypeList()) {
          ResourceTypeVisitor typeVisitor =
              pkgVisitor.enteringResourceType(
                  type.getTypeId().getId(), ResourceType.getEnum(type.getName()));
          if (typeVisitor != null) {
            for (Entry entry : type.getEntryList()) {
              ResourceValueVisitor entryVisitor =
                  typeVisitor.enteringDeclaration(entry.getName(), entry.getEntryId().getId());
              if (entryVisitor != null) {
                for (ConfigValue configValue : entry.getConfigValueList()) {
                  if (configValue.hasValue()) {
                    visitValue(entryVisitor, configValue.getValue(), sourcePool);
                  }
                }
              }
            }
          }
        }
      }
    }
    return visitor;
  }

  /** Accessor for the underlying URI of the apk. */
  public URI asApk() {
    return uri.normalize();
  }

  // TODO(72324748): Centralize duplicated code with AndroidCompiledDataDeserializer.
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

  private void visitValue(ResourceValueVisitor entryVisitor, Value value, List<String> sourcePool) {
    if (value.hasSource()) {
      entryVisitor.entering(apkFileSystem.getPath(sourcePool.get(value.getSource().getPathIdx())));
    }
    switch (value.getValueCase()) {
      case ITEM:
        visitItem(entryVisitor, value.getItem());
        break;
      case COMPOUND_VALUE:
        visitCompoundValue(entryVisitor, value.getCompoundValue());
        break;
      default:
        throw new IllegalStateException(
            "Config value does not have a declared value case: " + value);
    }
  }

  private void visitCompoundValue(ResourceValueVisitor entryVisitor, CompoundValue value) {
    switch (value.getValueCase()) {
      case STYLE:
        visitStyle(entryVisitor, value);
        break;
      case STYLEABLE:
        visitStyleable(entryVisitor, value);
        break;
      case ATTR:
        visitAttr(entryVisitor, value);
        break;
      case ARRAY:
        visitArray(entryVisitor, value);
        break;
      case PLURAL:
        visitPlural(entryVisitor, value);
        break;
      default:
    }
  }

  private void visitPlural(ResourceValueVisitor entryVisitor, CompoundValue value) {
    value.getPlural().getEntryList().stream()
        .filter(Plural.Entry::hasItem)
        .map(Plural.Entry::getItem)
        .forEach(
            i -> {
              switch (i.getValueCase()) {
                case FILE:
                  visitFile(entryVisitor, i.getFile());
                  break;
                case REF:
                  visitReference(entryVisitor, i.getRef());
                  break;
                default:
              }
            });
  }

  private void visitArray(ResourceValueVisitor entryVisitor, CompoundValue value) {
    value.getArray().getElementList().stream()
        .filter(Array.Element::hasItem)
        .map(Array.Element::getItem)
        .forEach(
            i -> {
              switch (i.getValueCase()) {
                case FILE:
                  visitFile(entryVisitor, i.getFile());
                  break;
                case REF:
                  visitReference(entryVisitor, i.getRef());
                  break;
                default:
              }
            });
  }

  private void visitAttr(ResourceValueVisitor entryVisitor, CompoundValue value) {
    value.getAttr().getSymbolList().stream()
        .filter(Symbol::hasName)
        .map(Symbol::getName)
        .forEach(name -> visitReference(entryVisitor, name));
  }

  private void visitStyleable(ResourceValueVisitor entryVisitor, CompoundValue value) {
    value.getStyleable().getEntryList().forEach(e -> visitReference(entryVisitor, e.getAttr()));
  }

  private void visitStyle(ResourceValueVisitor entryVisitor, CompoundValue value) {
    final Style style = value.getStyle();
    if (style.hasParent()) {
      visitReference(entryVisitor, style.getParent());
    }
    for (Style.Entry entry : style.getEntryList()) {
      if (entry.hasItem()) {
        visitItem(entryVisitor, entry.getItem());
      }
      if (entry.hasKey()) {
        visitReference(entryVisitor, entry.getKey());
      }
    }
  }

  private void visitItem(ResourceValueVisitor entryVisitor, Item item) {
    switch (item.getValueCase()) {
      case FILE:
        visitFile(entryVisitor, item.getFile());
        break;
      case REF:
        visitReference(entryVisitor, item.getRef());
        break;
      default:
    }
  }

  private void visitFile(ResourceValueVisitor entryVisitor, FileReference file) {
    final Path path = apkFileSystem.getPath(file.getPath());
    if (file.getType() == FileReference.Type.PROTO_XML) {
      visitXmlResource(path, entryVisitor.entering(path));
    } else if (file.getType() != FileReference.Type.PNG) {
      entryVisitor.acceptOpaqueFileType(path);
    }
  }

  private void visitXmlResource(Path path, ReferenceVisitor visitor) {
    if (visitor == null) {
      return;
    }

    try (InputStream in = Files.newInputStream(path)) {
      visit(XmlNode.parseFrom(in, ExtensionRegistry.getEmptyRegistry()), visitor);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private void visit(XmlNode node, ReferenceVisitor visitor) {
    if (node.hasElement()) {
      final XmlElement element = node.getElement();
      Consumer<XmlNode> visit =
          "wearableApp".equals(element.getName())
              ? n -> visitWearableApp(n, visitor)
              : n -> visit(n, visitor);

      visitAttributes(visitor, element);

      element.getChildList().forEach(visit);
    }
  }

  private void visitAttributes(ReferenceVisitor visitor, XmlElement element) {
    for (XmlAttribute attribute : element.getAttributeList()) {
      if (attribute.getCompiledItem().hasRef()) {
        visitReference(visitor, attribute.getCompiledItem().getRef());
      }
      if (attribute.getResourceId() != 0) {
        visitor.accept(attribute.getResourceId());
      }
    }
  }

  // TODO(b/113166518): Remove when wearable apps have real references.
  // this doesn't belong in the protoapk, but has to be included for the time being.
  private void visitWearableApp(XmlNode node, ReferenceVisitor visitor) {
    if (node.hasElement()) {
      final XmlElement element = node.getElement();
      if ("rawPathResId".equals(element.getName())) {
        visitor.accept(
            ResourceType.RAW.getName()
                + "/"
                + element.getChildList().stream().map(XmlNode::getText).collect(joining()));
      } else {
        visitAttributes(visitor, element);
      }
      element.getChildList().forEach(c -> visitWearableApp(c, visitor));
    }
  }

  private void visitReference(ReferenceVisitor visitor, Reference ref) {
    if (ref.getId() != 0) {
      logger.finest(
          "Visiting ref by id " + ref.getName() + "=" + "0x" + Integer.toHexString(ref.getId()));
      visitor.accept(ref.getId());
    } else if (!ref.getName().isEmpty()) {
      logger.finest("Visiting ref by name " + ref);
      visitor.accept(ref.getName());
    } else {
      logger.finest("Visiting null by name " + ref);
      visitor.acceptNullReference();
    }
  }

  @Override
  public void close() throws IOException {
    apkFileSystem.close();
  }

  /** Provides an entry point to recording declared and referenced resources in the apk. */
  public interface ResourceVisitor {
    /** Called when entering the manifest. If null, the manifest is not visited. */
    @Nullable
    ManifestVisitor enteringManifest();

    /** Called when entering a resource package. If null, the package is not visited. */
    @Nullable
    ResourcePackageVisitor enteringPackage(int pkgId, String packageName);
  }

  /** Provides a visitor for packages. */
  public interface ResourcePackageVisitor {
    /** Called when entering the resource types of the package. If null, the type is not visited. */
    @Nullable
    ResourceTypeVisitor enteringResourceType(int typeId, ResourceType type);
  }

  /** Visitor for resources types */
  public interface ResourceTypeVisitor {
    /**
     * Called for resource declarations.
     *
     * @param name The name of the resource.
     * @param resourceId The id of the resource, without the package and type.
     * @return A visitor for accepting references to other resources from the declared resource. If
     *     null, the value is not visited.
     */
    @Nullable
    ResourceValueVisitor enteringDeclaration(String name, int resourceId);
  }

  /** A manifest specific resource reference visitor. */
  public interface ManifestVisitor extends ReferenceVisitor {}

  /** General resource reference visitor. */
  public interface ResourceValueVisitor extends ReferenceVisitor {
    /** Called when entering the source of a value. Maybe called multiple times for each value. */
    ReferenceVisitor entering(Path path);

    /* Called when a raw resource contains a non-proto xml file type. */
    void acceptOpaqueFileType(Path path);
  }

  /** Role interface for visiting resource references. */
  public interface ReferenceVisitor {
    /** Called when a reference is defined by name (resourceType/name). */
    void accept(String name);

    /** Called when a reference is defined by id (full id, with package and type.) */
    void accept(int value);

    /** Called when a reference is null. */
    default void acceptNullReference() {
      // pass
    }
  }
}
