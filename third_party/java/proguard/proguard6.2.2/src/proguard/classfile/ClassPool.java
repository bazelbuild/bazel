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
package proguard.classfile;

import proguard.classfile.visitor.*;
import proguard.util.*;

import java.util.*;

/**
 * This is a set of representations of classes. They can be enumerated or
 * retrieved by name. They can also be accessed by means of class visitors.
 *
 * @author Eric Lafortune
 */
public class ClassPool
{
    // We're using a sorted tree map instead of a hash map to store the classes,
    // in order to make the processing more deterministic.
    private final Map classes = new TreeMap();


    /**
     * Clears the class pool.
     */
    public void clear()
    {
        classes.clear();
    }


    /**
     * Adds the given Clazz to the class pool.
     */
    public void addClass(Clazz clazz)
    {
        classes.put(clazz.getName(), clazz);
    }


    /**
     * Removes the given Clazz from the class pool.
     */
    public void removeClass(Clazz clazz)
    {
        removeClass(clazz.getName());
    }


    /**
     * Removes the specified Clazz from the class pool.
     */
    public void removeClass(String className)
    {
        classes.remove(className);
    }


    /**
     * Returns a Clazz from the class pool based on its name. Returns
     * <code>null</code> if the class with the given name is not in the class
     * pool.
     */
    public Clazz getClass(String className)
    {
        return (Clazz)classes.get(className);
    }


    /**
     * Returns an Iterator of all class names in the class pool.
     */
    public Iterator classNames()
    {
        return classes.keySet().iterator();
    }


    /**
     * Returns the number of classes in the class pool.
     */
    public int size()
    {
        return classes.size();
    }


    /**
     * Applies the given ClassPoolVisitor to the class pool.
     */
    public void accept(ClassPoolVisitor classPoolVisitor)
    {
        classPoolVisitor.visitClassPool(this);
    }


    /**
     * Applies the given ClassVisitor to all classes in the class pool,
     * in random order.
     */
    public void classesAccept(ClassVisitor classVisitor)
    {
        Iterator iterator = classes.values().iterator();
        while (iterator.hasNext())
        {
            Clazz clazz = (Clazz)iterator.next();
            clazz.accept(classVisitor);
        }
    }


    /**
     * Applies the given ClassVisitor to all classes in the class pool,
     * in sorted order.
     */
    public void classesAcceptAlphabetically(ClassVisitor classVisitor)
    {
        // We're already using a tree map.
        //TreeMap sortedClasses = new TreeMap(classes);
        //Iterator iterator = sortedClasses.values().iterator();

        Iterator iterator = classes.values().iterator();
        while (iterator.hasNext())
        {
            Clazz clazz = (Clazz)iterator.next();
            clazz.accept(classVisitor);
        }
    }


    /**
     * Applies the given ClassVisitor to all matching classes in the class pool.
     */
    public void classesAccept(String        classNameFilter,
                              ClassVisitor  classVisitor)
    {
        classesAccept(new ListParser(new ClassNameParser()).parse(classNameFilter),
                      classVisitor);
    }


    /**
     * Applies the given ClassVisitor to all matching classes in the class pool.
     */
    public void classesAccept(List          classNameFilter,
                              ClassVisitor  classVisitor)
    {
        classesAccept(new ListParser(new ClassNameParser()).parse(classNameFilter),
                      classVisitor);
    }


    /**
     * Applies the given ClassVisitor to all matching classes in the class pool.
     */
    public void classesAccept(StringMatcher classNameFilter,
                              ClassVisitor  classVisitor)
    {
        Iterator iterator = classes.entrySet().iterator();
        while (iterator.hasNext())
        {
            Map.Entry entry     = (Map.Entry)iterator.next();
            String    className = (String   )entry.getKey();

            if (classNameFilter.matches(className))
            {
                Clazz clazz = (Clazz)entry.getValue();
                clazz.accept(classVisitor);
            }
        }
    }


    /**
     * Applies the given ClassVisitor to the class with the given name,
     * if it is present in the class pool.
     */
    public void classAccept(String className, ClassVisitor classVisitor)
    {
        Clazz clazz = getClass(className);
        if (clazz != null)
        {
            clazz.accept(classVisitor);
        }
    }
}
