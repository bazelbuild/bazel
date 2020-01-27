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

package com.google.devtools.build.android.incrementaldeployment;

import android.app.Application;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.util.ArrayMap;
import android.util.Log;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.ref.WeakReference;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

  private static final FilenameFilter SO = new FilenameFilter() {
      public boolean accept(File dir, String name) {
        return name.endsWith(".so");
      }
    };

  private final String realClassName;
  private final String packageName;

  private String externalResourceFile;
  private Application realApplication;

  private Object stashedContentProviders;

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
    String dexDirectory = INCREMENTAL_DEPLOYMENT_DIR + "/" + packageName + "/dex";
    File[] dexes = new File(dexDirectory).listFiles();
    if (dexes == null) {
      throw new IllegalStateException(".dex directory '" + dexDirectory + "' does not exist");
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

  @SuppressWarnings("unchecked")
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

      if (android.os.Build.VERSION.SDK_INT <= android.os.Build.VERSION_CODES.KITKAT) {
        // Kitkat needs this method call, Lollipop doesn't.
        //
        // This method call was removed from Pie:
        // https://android.googlesource.com/platform/frameworks/base/+/bebfcc46a249a70af04bc18490a897888a142fb8%5E%21/#F7
        Method mEnsureStringBlocks = AssetManager.class.getDeclaredMethod("ensureStringBlocks");
        mEnsureStringBlocks.setAccessible(true);
        mEnsureStringBlocks.invoke(newAssetManager);
      }

      // Find the singleton instance of ResourcesManager
      Class<?> clazz = Class.forName("android.app.ResourcesManager");
      Method mGetInstance = clazz.getDeclaredMethod("getInstance");
      mGetInstance.setAccessible(true);
      Object resourcesManager = mGetInstance.invoke(null);

      // Get all known Resources objects
      Collection<WeakReference<Resources>> references;
      try {
        // Pre-N
        Field fMActiveResources = clazz.getDeclaredField("mActiveResources");
        fMActiveResources.setAccessible(true);
        ArrayMap<?, WeakReference<Resources>> arrayMap =
            (ArrayMap<?, WeakReference<Resources>>) fMActiveResources.get(resourcesManager);
        references = arrayMap.values();
      } catch (NoSuchFieldException e) {
        // N moved the resources to mResourceReferences
        Field mResourceReferences = clazz.getDeclaredField("mResourceReferences");
        mResourceReferences.setAccessible(true);
        references =
            (Collection<WeakReference<Resources>>) mResourceReferences.get(resourcesManager);
      }

      // Iterate over all known Resources objects
      for (WeakReference<Resources> wr : references) {
        Resources resources = wr.get();
        // Set the AssetManager of the Resources instance to our brand new one
        try {
          // Pre-N
          Field mAssets = Resources.class.getDeclaredField("mAssets");
          mAssets.setAccessible(true);
          mAssets.set(resources, newAssetManager);
        } catch (NoSuchFieldException e) {
          // N moved the mAssets inside an mResourcesImpl field
          Field mResourcesImplField = Resources.class.getDeclaredField("mResourcesImpl");
          mResourcesImplField.setAccessible(true);
          Object mResourceImpl = mResourcesImplField.get(resources);
          Field implAssets = mResourceImpl.getClass().getDeclaredField("mAssets");
          implAssets.setAccessible(true);
          implAssets.set(mResourceImpl, newAssetManager);
        }
        resources.updateConfiguration(resources.getConfiguration(), resources.getDisplayMetrics());
      }
    } catch (IllegalAccessException | NoSuchFieldException | NoSuchMethodException |
        ClassNotFoundException | InvocationTargetException | InstantiationException e) {
      throw new IllegalStateException(e);
    }
  }

  private void instantiateRealApplication(File codeCacheDir, String dataDir) {
    externalResourceFile = getExternalResourceFile();

    String nativeLibDir;
    try {
      // We cannot use the .so files pushed by adb for some reason: even if permissions are 777
      // and they are chowned to the user of the app from a root shell, dlopen() returns with
      // "Permission denied". For some reason, copying them over makes them work (at the cost of
      // some execution time and complexity here, of course)
      nativeLibDir = copyNativeLibs(dataDir);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }

    IncrementalClassLoader.inject(
        StubApplication.class.getClassLoader(),
        packageName,
        codeCacheDir,
        nativeLibDir,
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

  private String copyNativeLibs(String dataDir) throws IOException {
    File nativeLibDir = new File(INCREMENTAL_DEPLOYMENT_DIR + "/" + packageName + "/native");
    File newManifestFile = new File(nativeLibDir, "native_manifest");
    File incrementalDir = new File(dataDir + "/incrementallib");
    File installedManifestFile = new File(incrementalDir, "manifest");
    String defaultNativeLibDir = dataDir + "/lib";

    if (!newManifestFile.exists()) {
      // Native libraries are not installed incrementally. Just use the regular directory.
      return defaultNativeLibDir;
    }

    Map<String, String> newManifest = parseManifest(newManifestFile);
    Map<String, String> installedManifest = new LinkedHashMap<String, String>();
    Set<String> libsToDelete = new LinkedHashSet<String>();
    Set<String> libsToUpdate = new LinkedHashSet<String>();

    String realNativeLibDir = newManifest.isEmpty()
        ? defaultNativeLibDir : incrementalDir.toString();

    if (!incrementalDir.exists()) {
      if (!incrementalDir.mkdirs()) {
        throw new IOException("Could not mkdir " + incrementalDir);
      }
    }

    if (installedManifestFile.exists()) {
      installedManifest = parseManifest(installedManifestFile);
    } else {
      // Delete old libraries, in case things got out of sync.
      for (String installed : incrementalDir.list(SO)) {
        libsToDelete.add(installed);
      }
    }

    for (String installed : installedManifest.keySet()) {
      if (!newManifest.containsKey(installed)
          || !newManifest.get(installed).equals(installedManifest.get(installed))) {
        libsToDelete.add(installed);
      }
    }

    for (String newLib : newManifest.keySet()) {
      if (!installedManifest.containsKey(newLib)
          || !installedManifest.get(newLib).equals(newManifest.get(newLib))) {
        libsToUpdate.add(newLib);
      }
    }

    if (libsToDelete.isEmpty() && libsToUpdate.isEmpty()) {
      // Nothing to be done. Be lazy.
      return realNativeLibDir;
    }

    // Delete the installed manifest file. If anything below goes wrong, everything will be
    // reinstalled the next time the app starts up.
    installedManifestFile.delete();

    for (String toDelete : libsToDelete) {
      File fileToDelete = new File(incrementalDir + "/" + toDelete);
      Log.v("StubApplication", "Deleting " + fileToDelete);
      if (fileToDelete.exists() && !fileToDelete.delete()) {
        throw new IOException("Could not delete " + fileToDelete);
      }
    }

    for (String toUpdate : libsToUpdate) {
      Log.v("StubApplication", "Copying: " + toUpdate);
      File src = new File(nativeLibDir + "/" + toUpdate);
      copy(src, new File(incrementalDir + "/" + toUpdate));
    }

    try {
      copy(newManifestFile, installedManifestFile);
    } finally {
      // If we can't write the installed manifest file, delete it completely so that the next
      // time we get here we can start with a clean slate.
      installedManifestFile.delete();
    }

    return realNativeLibDir;
  }

  private static Map<String, String> parseManifest(File file) throws IOException {
    Map<String, String> result = new LinkedHashMap<>();
    try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
      while (true) {
        String line = reader.readLine();
        if (line == null) {
          break;
        }

        String[] items = line.split(" ");
        result.put(items[0], items[1]);
      }
    }

    return result;
  }


  private static void copy(File src, File dst) throws IOException {
    Log.v("StubApplication", "Copying " + src + " -> " + dst);
    InputStream in = null;
    OutputStream out = null;
    try {
      in = new FileInputStream(src);
      out = new FileOutputStream(dst);

      // Transfer bytes from in to out
      byte[] buf = new byte[1048576];
      int len;
      while ((len = in.read(buf)) > 0) {
        out.write(buf, 0, len);
      }
    } finally {
      if (in != null) {
        in.close();
      }

      if (out != null) {
        out.close();
      }
    }
  }

  private static Field getField(Object instance, String fieldName)
      throws ClassNotFoundException {
    for (Class<?> clazz = instance.getClass(); clazz != null; clazz = clazz.getSuperclass()) {
      try {
        Field field = clazz.getDeclaredField(fieldName);
        field.setAccessible(true);
        return field;
      } catch (NoSuchFieldException e) {
        // IllegalStateException will be thrown below
      }
    }

    throw new IllegalStateException("Field '" + fieldName + "' not found");
  }

  private void enableContentProviders() {
    Log.v("INCREMENTAL", "enableContentProviders");
    try {
      Class<?> activityThread = Class.forName("android.app.ActivityThread");
      Method mCurrentActivityThread = activityThread.getMethod("currentActivityThread");
      mCurrentActivityThread.setAccessible(true);
      Object currentActivityThread = mCurrentActivityThread.invoke(null);
      Object boundApplication = getField(
          currentActivityThread, "mBoundApplication").get(currentActivityThread);
      getField(boundApplication, "providers").set(boundApplication, stashedContentProviders);
      if (stashedContentProviders != null) {
        Method mInstallContentProviders = activityThread.getDeclaredMethod(
            "installContentProviders", Context.class, List.class);
        mInstallContentProviders.setAccessible(true);
        mInstallContentProviders.invoke(
            currentActivityThread, realApplication, stashedContentProviders);
        stashedContentProviders = null;
      }
    } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException
        | InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
  }

  // ActivityThread instantiates all the content providers between attachBaseContext() and
  // onCreate(). Since we replace the Application instance in onCreate(), this may fail if
  // they depend on the correct Application being present, so we postpone instantiating the
  // content providers until we have the real Application instance.
  private void disableContentProviders() {
    Log.v("INCREMENTAL", "disableContentProviders");
    try {
      Class<?> activityThread = Class.forName("android.app.ActivityThread");
      Method mCurrentActivityThread = activityThread.getMethod("currentActivityThread");
      mCurrentActivityThread.setAccessible(true);
      Object currentActivityThread = mCurrentActivityThread.invoke(null);
      Object boundApplication = getField(
          currentActivityThread, "mBoundApplication").get(currentActivityThread);
      Field fProviders = getField(boundApplication, "providers");

      stashedContentProviders = fProviders.get(boundApplication);
      fProviders.set(boundApplication, null);
    } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException
        | InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  protected void attachBaseContext(Context context) {
    instantiateRealApplication(
        context.getCacheDir(),
        context.getApplicationInfo().dataDir);

    // This is called from ActivityThread#handleBindApplication() -> LoadedApk#makeApplication().
    // Application#mApplication is changed right after this call, so we cannot do the monkey
    // patching here. So just forward this method to the real Application instance.
    super.attachBaseContext(context);

    try {
      Method attachBaseContext =
          ContextWrapper.class.getDeclaredMethod("attachBaseContext", Context.class);
      attachBaseContext.setAccessible(true);
      attachBaseContext.invoke(realApplication, context);
      disableContentProviders();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public void onCreate() {
    monkeyPatchApplication();
    monkeyPatchExistingResources();
    enableContentProviders();
    super.onCreate();
    realApplication.onCreate();
  }
}
