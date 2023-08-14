/*
 * Copyright (C) 2011 The Android Open Source Project
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

public final class SizeOf {
    private SizeOf() {}

    public static final int UBYTE = 1;
    public static final int USHORT = 2;
    public static final int UINT = 4;

    public static final int SIGNATURE = UBYTE * 20;

    /**
     * magic ubyte[8]
     * checksum uint
     * signature ubyte[20]
     * file_size uint
     * header_size uint
     * endian_tag uint
     * link_size uint
     * link_off uint
     * map_off uint
     * string_ids_size uint
     * string_ids_off uint
     * type_ids_size uint
     * type_ids_off uint
     * proto_ids_size uint
     * proto_ids_off uint
     * field_ids_size uint
     * field_ids_off uint
     * method_ids_size uint
     * method_ids_off uint
     * class_defs_size uint
     * class_defs_off uint
     * data_size uint
     * data_off uint
     */
    public static final int HEADER_ITEM = (8 * UBYTE) + UINT + SIGNATURE + (20 * UINT); // 0x70

    /**
     * string_data_off uint
     */
    public static final int STRING_ID_ITEM = UINT;

    /**
     * descriptor_idx uint
     */
    public static final int TYPE_ID_ITEM = UINT;

    /**
     * type_idx ushort
     */
    public static final int TYPE_ITEM = USHORT;

    /**
     * shorty_idx uint
     * return_type_idx uint
     * return_type_idx uint
     */
    public static final int PROTO_ID_ITEM = UINT + UINT + UINT;

    /**
     * class_idx ushort
     * type_idx/proto_idx ushort
     * name_idx uint
     */
    public static final int MEMBER_ID_ITEM = USHORT + USHORT + UINT;

    /**
     * class_idx uint
     * access_flags uint
     * superclass_idx uint
     * interfaces_off uint
     * source_file_idx uint
     * annotations_off uint
     * class_data_off uint
     * static_values_off uint
     */
    public static final int CLASS_DEF_ITEM = 8 * UINT;

    /**
     * type ushort
     * unused ushort
     * size uint
     * offset uint
     */
    public static final int MAP_ITEM = USHORT + USHORT + UINT + UINT;

    /**
     * start_addr uint
     * insn_count ushort
     * handler_off ushort
     */
    public static final int TRY_ITEM = UINT + USHORT + USHORT;

    /**
     * call_site_off uint
     */
    public static final int CALL_SITE_ID_ITEM = UINT;

    /**
     * method_handle_type ushort
     * unused ushort
     * field_or_method_id ushort
     * unused ushort
     */
    public static final int METHOD_HANDLE_ITEM = USHORT + USHORT + USHORT + USHORT;
}
