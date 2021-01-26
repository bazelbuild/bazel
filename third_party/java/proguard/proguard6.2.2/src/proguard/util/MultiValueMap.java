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
package proguard.util;


import java.util.*;

/**
 * A key-values map that can have multiple values associated with each key.
 *
 * There is an efficient lookup method to retrieve all values of all keys.
 *
 * @param <K> the key type
 * @param <V> the value type
 *
 * @author Johan Leys
 */
public class MultiValueMap<K, V>
{
    private final Map<K, Set<V>> keyValueMap = new HashMap<K, Set<V>>();

    private final Set<V> values = new HashSet<V>();


    public void put(K key, V value)
    {
        putAll(key, Collections.singleton(value));
    }


    public void putAll(Set<K> key, V value)
    {
        putAll(key, Collections.singleton(value));
    }


    public void putAll(Set<K> keys, Set<V> values)
    {
        for (K key : keys)
        {
            putAll(key, values);
        }
    }


    public void putAll(K key, Set<V> values)
    {
        this.values.addAll(values);
        Set<V> existingValues = keyValueMap.get(key);
        if (existingValues == null)
        {
            existingValues = new HashSet<V>();
            keyValueMap.put(key, existingValues);
        }
        existingValues.addAll(values);
    }


    public Set<V> get(K key)
    {
        return keyValueMap.get(key);
    }


    /**
     * Returns a Set with all values of all keys.
     *
     * @return a Set with all values of all keys.
     */
    public Set<V> getValues()
    {
        return values;
    }
}
