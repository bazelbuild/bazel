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
package com.google.devtools.build.android;

import static com.google.common.base.Predicates.not;

import com.android.SdkConstants;
import com.google.common.base.MoreObjects;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.Optional;
import java.util.logging.Logger;
import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

/** Represents an AndroidManifest in the context of resource link and compilation. */
public class AndroidManifest {

  private static final Logger logger = Logger.getLogger(AndroidManifest.class.getName());

  private static final QName PACKAGE_NAME = QName.valueOf("package");
  private static final QName MANIFEST = QName.valueOf("manifest");
  private static final QName USES_SDK = QName.valueOf("uses-sdk");
  private static final QName MIN_SDK =
      new QName(SdkConstants.ANDROID_URI, "minSdkVersion", "android");
  private final String packageName;
  private final String minSdk;

  private AndroidManifest(String packageName, String minSdk) {
    this.packageName = packageName;
    this.minSdk = minSdk;
  }

  /** Parses the manifest from the path. */
  public static AndroidManifest parseFrom(Path manifest) {
    try (InputStream input = Files.newInputStream(manifest)) {
      final XMLEventReader xmlEventReader =
          XmlResourceValues.getXmlInputFactory().createXMLEventReader(input);

      String packageName = "";
      String minSdk = "";
      while (xmlEventReader.hasNext()) {
        final XMLEvent xmlEvent = xmlEventReader.nextEvent();
        if (xmlEvent.isStartElement()) {
          final StartElement element = xmlEvent.asStartElement();
          if (MANIFEST.equals(element.getName())) {
            packageName = getAttributeValue(xmlEvent, PACKAGE_NAME).orElse(packageName);
          }
          if (USES_SDK.equals(element.getName())) {
            minSdk = getAttributeValue(xmlEvent, MIN_SDK).orElse(minSdk);
          }
        }
      }

      if (minSdk.isEmpty()) {
        logger.warning(
            String.format(
                "\n\u001B[31mCONFIGURATION:\u001B[0m" + " %s has no minSdkVersion. Using 1.",
                manifest));
      }

      return new AndroidManifest(packageName, minSdk.isEmpty() ? "1" : minSdk);
    } catch (IOException | XMLStreamException e) {
      throw new ManifestProcessingException(e);
    }
  }

  private static Optional<String> getAttributeValue(XMLEvent xmlEvent, QName attributeName) {
    return Optional.ofNullable(xmlEvent.asStartElement().getAttributeByName(attributeName))
        .map(Attribute::getValue)
        .filter(not(String::isEmpty));
  }

  public static AndroidManifest asEmpty() {
    return new AndroidManifest("", "1");
  }

  public static AndroidManifest of(String packageName, String minSdk) {
    return new AndroidManifest(packageName, minSdk);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof AndroidManifest)) {
      return false;
    }
    AndroidManifest that = (AndroidManifest) o;
    return Objects.equals(packageName, that.packageName) && Objects.equals(minSdk, that.minSdk);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("packageName", packageName)
        .add("minSdk", minSdk)
        .toString();
  }

  @Override
  public int hashCode() {
    return Objects.hash(packageName, minSdk);
  }

  /** Creates a dummy manifest for processing without any manifest template variables. */
  public Path writeDummyManifestForAapt(Path workingDirectory, String packageForR) {
    try {
      return Files.write(
          Files.createDirectories(workingDirectory).resolve("AndroidManifest.xml"),
          ImmutableList.of(
              "<?xml version='1.0' encoding='utf-8'?>",
              "<manifest xmlns:android='http://schemas.android.com/apk/res/android'",
              String.format(
                  "package='%s'>", Strings.isNullOrEmpty(packageForR) ? packageName : packageForR),
              "<application/>",
              Strings.isNullOrEmpty(minSdk)
                  ? ""
                  : String.format("<uses-sdk android:minSdkVersion='%s'/>", minSdk),
              "</manifest>"),
          StandardOpenOption.CREATE_NEW);
    } catch (IOException e) {
      throw new AndroidManifestProcessor.ManifestProcessingException(e);
    }
  }
}
