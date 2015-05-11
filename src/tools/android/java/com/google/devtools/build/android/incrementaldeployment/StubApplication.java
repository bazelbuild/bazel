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

package com.google.devtools.build.android.incrementaldeployment;

import android.app.Application;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.util.ArrayMap;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import java.lang.ref.WeakReference;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A stub application that patches the class loader, then replaces itself with the real application
 * by applying a liberal amount of reflection on Android internals.
 *
 * <p>This is, of course, terribly error-prone. Most of this code was tested with API versions
 * 8, 10, 14, 15, 16, 17, 18, 19 and 21 on the Android emulator, a Nexus 5 running Lollipop LRX22C
 * and a Samsung GT-I5800 running Froyo XWJPE. The exception is {@code monkeyPatchAssetManagers},
 * which only works on Kitkat and Lollipop.
 *
 * <p>Note that due to a bug in Dalvik, this only works on Kitkat if ART is the Java runtime.
 *
 * <p>Unfortunately, if this does not work, we don't have a fallback mechanism: as soon as we
 * build the APK with this class as the Application, we are committed to going through with it.
 *
 * <p>This class should use as few other classes as possible before the class loader is patched
 * because any class loaded before it cannot be incrementally deployed.
 */
public class StubApplication extends Application {
  private static final String INCREMENTAL_DEPLOYMENT_DIR = "/data/local/tmp/incrementaldeployment";

  private final String realClassName;
  private final String packageName;

  private String externalResourceFile;
  private Application realApplication;

  public StubApplication() {
    String[] stubApplicationData = getResourceAsString("stub_application_data.txt").split("\n");
    realClassName = stubApplicationData[0];
    packageName = stubApplicationData[1];

    Log.v("StubApplication", String.format(
        "StubApplication created. Android package is %s, real application class is %s.",
        packageName, realClassName));
  }

  private String getExternalResourceFile() {
    String base = INCREMENTAL_DEPLOYMENT_DIR + "/" + packageName + "/";
    String resourceFile = base + "resources.ap_";
    if (!(new File(resourceFile).isFile())) {
      resourceFile = base + "resources";
      if (!(new File(resourceFile).isDirectory())) {
        Log.v("StubApplication", "Cannot find external resources, not patching them in");
        return null;
      }
    }

    Log.v("StubApplication", "Found external resources at " + resourceFile);
    return resourceFile;
  }

  private List<String> getDexList(String packageName) {
    List<String> result = new ArrayList<>();
    File[] dexes = new File(INCREMENTAL_DEPLOYMENT_DIR + "/" + packageName + "/dex").listFiles();
    if (dexes == null) {
      throw new IllegalStateException(".dex directory does not exist");
    }

    for (File dex : dexes) {
      if (dex.getName().endsWith(".dex")) {
        result.add(dex.getPath());
      }
    }

    return result;
  }

  private String getResourceAsString(String resource) {
    InputStream resourceStream = null;
    // try-with-resources would be much nicer, but that requires SDK level 19, and we want this code
    // to be compatible with earlier Android versions
    try {
      resourceStream = getClass().getClassLoader().getResourceAsStream(resource);
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      byte[] buffer = new byte[1024];
      int length = 0;
      while ((length = resourceStream.read(buffer)) != -1) {
        baos.write(buffer, 0, length);
      }

      String result = new String(baos.toByteArray(), "UTF-8");
      return result;
    } catch (IOException e) {
      throw new IllegalStateException(e);
    } finally {
      if (resourceStream != null) {
        try {
          resourceStream.close();
        } catch (IOException e) {
          // Not much we can do here
        }
      }
    }
  }

  @SuppressWarnings("unchecked")  // Lots of conversions with generic types
  private void monkeyPatchApplication() {
    // StubApplication is created by reflection in Application#handleBindApplication() ->
    // LoadedApk#makeApplication(), and its return value is used to set the Application field in all
    // sorts of Android internals.
    //
    // Fortunately, Application#onCreate() is called quite soon after, so what we do is monkey
    // patch in the real Application instance in StubApplication#onCreate().
    //
    // A few places directly use the created Application instance (as opposed to the fields it is
    // eventually stored in). Fortunately, it's easy to forward those to the actual real
    // Application class.
    try {
      // Find the ActivityThread instance for the current thread
      Class<?> activityThread = Class.forName("android.app.ActivityThread");
      Method m = activityThread.getMethod("currentActivityThread");
      m.setAccessible(true);
      Object currentActivityThread = m.invoke(null);

      // Find the mInitialApplication field of the ActivityThread to the real application
      Field mInitialApplication = activityThread.getDeclaredField("mInitialApplication");
      mInitialApplication.setAccessible(true);
      Application initialApplication = (Application) mInitialApplication.get(currentActivityThread);
      if (initialApplication == StubApplication.this) {
        mInitialApplication.set(currentActivityThread, realApplication);
      }

      // Replace all instance of the stub application in ActivityThread#mAllApplications with the
      // real one
      Field mAllApplications = activityThread.getDeclaredField("mAllApplications");
      mAllApplications.setAccessible(true);
      List<Application> allApplications = (List<Application>) mAllApplications
          .get(currentActivityThread);
      for (int i = 0; i < allApplications.size(); i++) {
        if (allApplications.get(i) == StubApplication.this) {
          allApplications.set(i, realApplication);
        }
      }

      // Figure out how loaded APKs are stored.

      // API version 8 has PackageInfo, 10 has LoadedApk. 9, I don't know.
      Class<?> loadedApkClass;
      try {
        loadedApkClass = Class.forName("android.app.LoadedApk");
      } catch (ClassNotFoundException e) {
        loadedApkClass = Class.forName("android.app.ActivityThread$PackageInfo");
      }
      Field mApplication = loadedApkClass.getDeclaredField("mApplication");
      mApplication.setAccessible(true);
      Field mResDir = loadedApkClass.getDeclaredField("mResDir");
      mResDir.setAccessible(true);

      // 10 doesn't have this field, 14 does. Fortunately, there are not many Honeycomb devices
      // floating around.
      Field mLoadedApk = null;
      try {
        mLoadedApk = Application.class.getDeclaredField("mLoadedApk");
      } catch (NoSuchFieldException e) {
        // According to testing, it's okay to ignore this.
      }

      // Enumerate all LoadedApk (or PackageInfo) fields in ActivityThread#mPackages and
      // ActivityThread#mResourcePackages and do two things:
      //   - Replace the Application instance in its mApplication field with the real one
      //   - Replace mResDir to point to the external resource file instead of the .apk. This is
      //     used as the asset path for new Resources objects.
      //   - Set Application#mLoadedApk to the found LoadedApk instance
      for (String fieldName : new String[] { "mPackages", "mResourcePackages" }) {
        Field field = activityThread.getDeclaredField(fieldName);
        field.setAccessible(true);
        Object value = field.get(currentActivityThread);

        for (Map.Entry<String, WeakReference<?>> entry :
            ((Map<String, WeakReference<?>>) value).entrySet()) {
          Object loadedApk = entry.getValue().get();
          if (loadedApk == null) {
            continue;
          }

          if (mApplication.get(loadedApk) == StubApplication.this) {
            mApplication.set(loadedApk, realApplication);
            if (externalResourceFile != null) {
              mResDir.set(loadedApk, externalResourceFile);
            }

            if (mLoadedApk != null) {
              mLoadedApk.set(realApplication, loadedApk);
            }
          }
        }
      }
    } catch (IllegalAccessException | NoSuchFieldException | NoSuchMethodException |
        ClassNotFoundException | InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
  }

  private void monkeyPatchExistingResources() {
    if (externalResourceFile == null) {
      return;
    }

    try {
      // Create a new AssetManager instance and point it to the resources installed under
      // /sdcard
      AssetManager newAssetManager = AssetManager.class.getConstructor().newInstance();
      Method mAddAssetPath = AssetManager.class.getDeclaredMethod("addAssetPath", String.class);
      mAddAssetPath.setAccessible(true);
      if (((int) mAddAssetPath.invoke(newAssetManager, externalResourceFile)) == 0) {
        throw new IllegalStateException("Could not create new AssetManager");
      }

      // Kitkat needs this method call, Lollipop doesn't. However, it doesn't seem to cause any harm
      // in L, so we do it unconditionally.
      Method mEnsureStringBlocks = AssetManager.class.getDeclaredMethod("ensureStringBlocks");
      mEnsureStringBlocks.setAccessible(true);
      mEnsureStringBlocks.invoke(newAssetManager);

      // Find the singleton instance of ResourcesManager
      Class<?> clazz = Class.forName("android.app.ResourcesManager");
      Method mGetInstance = clazz.getDeclaredMethod("getInstance");
      mGetInstance.setAccessible(true);
      Object resourcesManager = mGetInstance.invoke(null);

      Field mAssets = Resources.class.getDeclaredField("mAssets");
      mAssets.setAccessible(true);

      // Iterate over all known Resources objects
      Field fMActiveResources = clazz.getDeclaredField("mActiveResources");
      fMActiveResources.setAccessible(true);
      @SuppressWarnings("unchecked")
      ArrayMap<?, WeakReference<Resources>> arrayMap =
          (ArrayMap<?, WeakReference<Resources>>) fMActiveResources.get(resourcesManager);
      for (WeakReference<Resources> wr : arrayMap.values()) {
        Resources resources = wr.get();
        // Set the AssetManager of the Resources instance to our brand new one
        mAssets.set(resources, newAssetManager);
        resources.updateConfiguration(resources.getConfiguration(), resources.getDisplayMetrics());
      }
    } catch (IllegalAccessException | NoSuchFieldException | NoSuchMethodException |
        ClassNotFoundException | InvocationTargetException | InstantiationException e) {
      throw new IllegalStateException(e);
    }
  }

  private void instantiateRealApplication(String codeCacheDir) {
    externalResourceFile = getExternalResourceFile();

    IncrementalClassLoader.inject(
        StubApplication.class.getClassLoader(),
        packageName,
        codeCacheDir,
        getDexList(packageName));

    try {
      @SuppressWarnings("unchecked")
      Class<? extends Application> realClass =
          (Class<? extends Application>) Class.forName(realClassName);
      Constructor<? extends Application> ctor = realClass.getConstructor();
      realApplication = ctor.newInstance();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  protected void attachBaseContext(Context context) {
    instantiateRealApplication(context.getCacheDir().getPath());

    // This is called from ActivityThread#handleBindApplication() -> LoadedApk#makeApplication().
    // Application#mApplication is changed right after this call, so we cannot do the monkey
    // patching here. So just forward this method to the real Application instance.
    super.attachBaseContext(context);

    try {
      Method attachBaseContext =
          ContextWrapper.class.getDeclaredMethod("attachBaseContext", Context.class);
      attachBaseContext.setAccessible(true);
      attachBaseContext.invoke(realApplication, context);

    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public void onCreate() {
    monkeyPatchApplication();
    monkeyPatchExistingResources();
    super.onCreate();
    realApplication.onCreate();
  }
}
