/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.sdklib.build;

/**
 * An exception thrown when trying to add files to a sealed APK.
 */
public final class SealedApkException extends Exception {
    private static final long serialVersionUID = 1L;

    public SealedApkException(String format, Object... args) {
        super(String.format(format, args));
    }

    public SealedApkException(Throwable cause, String format, Object... args) {
        super(String.format(format, args), cause);
    }

    public SealedApkException(Throwable cause) {
        super(cause);
    }
}