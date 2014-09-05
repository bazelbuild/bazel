/*
 * Copyright 2013-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.facebook.buck.apple.xcode;

import com.dd.plist.NSArray;
import com.dd.plist.NSDictionary;
import com.dd.plist.NSObject;
import com.dd.plist.NSString;
import com.facebook.buck.apple.xcode.xcodeproj.PBXObject;
import com.facebook.buck.apple.xcode.xcodeproj.PBXProject;

import java.util.List;

import javax.annotation.concurrent.NotThreadSafe;

/**
 * Serializer that handles conversion of an in-memory object graph representation of an xcode
 * project (instances of {@link PBXObject}) into an Apple property list.
 *
 * Serialization proceeds from the root object, a ${link PBXProject} instance, to all of its
 * referenced objects. Each object being visited calls back into this class ({@link #addField}) to
 * populate the plist representation with its fields.
 */
@NotThreadSafe
public class XcodeprojSerializer {
  private final PBXProject rootObject;
  private final NSDictionary objects;
  private final GidGenerator gidGenerator;
  private NSDictionary currentObject;

  public XcodeprojSerializer(GidGenerator gidGenerator, PBXProject project) {
    rootObject = project;
    objects = new NSDictionary();
    this.gidGenerator = gidGenerator;
  }

  /**
   * Generate a plist serialization of project bound to this serializer.
   */
  public NSDictionary toPlist() {
    serializeObject(rootObject);

    NSDictionary root = new NSDictionary();
    root.put("archiveVersion", "1");
    root.put("classes", new NSDictionary());
    root.put("objectVersion", "46");
    root.put("objects", objects);
    root.put("rootObject", rootObject.getGlobalID());

    return root;
  }

  /**
   * Serialize a {@link PBXObject} and its recursive descendants into the object dictionary.
   *
   * @return the GID of the serialized object
   *
   * @see PBXObject#serializeInto
   */
  private String serializeObject(PBXObject obj) {
    if (obj.getGlobalID() == null) {
      obj.setGlobalID(obj.generateGid(gidGenerator));
    } else {
      // Check that the object has already been serialized.
      NSObject object = objects.get(obj.getGlobalID());
      if (object != null) {
        return obj.getGlobalID();
      }
    }

    // Save the existing object being deserialized.
    NSDictionary stack = currentObject;

    currentObject = new NSDictionary();
    currentObject.put("isa", obj.isa());
    obj.serializeInto(this);
    objects.put(obj.getGlobalID(), currentObject);

    // Restore the existing object being deserialized.
    currentObject = stack;
    return obj.getGlobalID();
  }

  public void addField(String name, PBXObject obj) {
    if (obj != null) {
      String gid = serializeObject(obj);
      currentObject.put(name, gid);
    }
  }

  public void addField(String name, int val) {
    currentObject.put(name, val);
  }

  public void addField(String name, String val) {
    if (val != null) {
      currentObject.put(name, val);
    }
  }

  public void addField(String name, boolean val) {
    currentObject.put(name, val);
  }

  public void addField(String name, List<? extends PBXObject> objectList) {
    NSArray array = new NSArray(objectList.size());
    for (int i = 0; i < objectList.size(); i++) {
      String gid = serializeObject(objectList.get(i));
      array.setValue(i, new NSString(gid));
    }
    currentObject.put(name, array);
  }

  public void addField(String name, NSObject v) {
    if (v != null) {
      currentObject.put(name, v);
    }
  }
}
