/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.optimize.gson;

import java.util.*;

/**
 * This class keeps track of which Java classes and fields can be involved in
 * Json (de)serialization and stores their corresponding Json field names
 * and internal indices.
 *
 * @author Lars Vandenbergh
 * @author Rob Coekaerts
 */
public class OptimizedJsonInfo
{
    /**
     * Maps the class name to a unique, contiguous index.
     * This index is used to generate unique suffixes for the generated toJson
     * and fromJson methods.
     */
    public Map<String, Integer> classIndices = new HashMap<String, Integer>();

    /**
     * Maps the json field name to a unique, contiguous index.
     * This index is used to map Json field names to indices and backwards
     * in the _OptimizedJsonReader and _OptimizedJsonWriter.
     */
    public Map<String, Integer> jsonFieldIndices = new HashMap<String, Integer>();

    /**
     * Maps the class name to a ClassJsonInfo. The ClassJsonInfo contains the
     * names of all the exposed Java fields and their corresponding Json field
     * name(s).
     */
    public Map<String,ClassJsonInfo> classJsonInfos = new HashMap<String, ClassJsonInfo>();


    /**
     * Assigns indices to all registered classes and fields.
     * The generated indices will be contiguous and starting from 0, both
     * for classes and fields.
     */
    public void assignIndices()
    {
        assignIndices(classIndices);
        assignIndices(jsonFieldIndices);
    }

    private void assignIndices(Map<String, Integer> indexMap)
    {
        int index = 0;
        for (String fieldName : indexMap.keySet())
        {
            indexMap.put(fieldName, index++);
        }
    }

    public static class ClassJsonInfo
    {
        /**
         * Maps the Java field name to all of its corresponding Json field names.
         * The first name in the array is the primary name that is used for
         * writing to Json. The remaining names are alternatives that are
         * also accepted when reading from Json.
         */
        public Map<String, String[]> javaToJsonFieldNames  = new HashMap<String, String[]>();

        /**
         * Contains the names of all (and only those) Java fields that are
         * exposed.
         */
        public Set<String>           exposedJavaFieldNames = new HashSet<String>();
    }
}
