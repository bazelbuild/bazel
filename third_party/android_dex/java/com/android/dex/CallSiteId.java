/*
 * Copyright (C) 2017 The Android Open Source Project
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

package com.android.dex;

import com.android.dex.Dex.Section;
import com.android.dex.util.Unsigned;

/**
 * A call_site_id_item: https://source.android.com/devices/tech/dalvik/dex-format#call-site-id-item
 */
public class CallSiteId implements Comparable<CallSiteId> {

    private final Dex dex;
    private final int offset;

    public CallSiteId(Dex dex, int offset) {
        this.dex = dex;
        this.offset = offset;
    }

    @Override
    public int compareTo(CallSiteId o) {
        return Unsigned.compare(offset, o.offset);
    }

    public int getCallSiteOffset() {
        return offset;
    }

    public void writeTo(Section out) {
        out.writeInt(offset);
    }

    @Override
    public String toString() {
        if (dex == null) {
            return String.valueOf(offset);
        }
        return dex.protoIds().get(offset).toString();
    }
}
